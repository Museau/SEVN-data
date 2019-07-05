""" pre-process and write the labels, spatial graph, and lower resolution images to disk """
from __future__ import print_function, division
import glob
import os
import xml.etree.ElementTree as et
from tqdm import tqdm
import networkx as nx
import pandas as pd
import numpy as np
import cv2
import gzip
import pickle
from matplotlib import pyplot as plt

height = 126
width = 224
shape = (3840, 2160)
crop_margin = int(height * (1/6))


def process_labels(paths):
    """ This function processes the labels into a nice format for the simulator"""
    labels = []
    failed_to_parse = []
    for p in paths:
        xtree = et.parse(p)
        xroot = xtree.getroot()
        for idx, node in enumerate(xroot):
            if node.tag != "object":
                continue
            frame = int(p.split("_")[-1].split(".")[0])
            text_label = node.find("name").text
            house_number = None
            if text_label.split("-")[0] == "street_sign":
                try:
                    obj_type, street_name = text_label.split("-")
                except Exception as e:
                    print("street_sign: " + str(e))
                    failed_to_parse.append(text_label)
                    continue
            elif text_label.split("-")[0] == "house_number":
                try:
                    obj_type, house_number, street_name = text_label.split("-")
                except Exception as e:
                    print("house_number: " + str(e))
                    failed_to_parse.append(text_label)
                    continue
            elif text_label.split("-")[0] == "door":
                try:
                    obj_type, house_number, street_name = text_label.split("-")
                except Exception as e:
                    print("door: " + str(e))
                    failed_to_parse.append(text_label)
                    continue
            x_min = int(width * int(node.find("bndbox").find("xmin").text) / shape[0])
            x_max = int(width * int(node.find("bndbox").find("xmax").text) / shape[0])
            y_min = int((height - 2 * crop_margin) * int(node.find("bndbox").find("ymin").text)/ shape[1])
            y_max = int((height - 2 * crop_margin) * int(node.find("bndbox").find("ymax").text) / shape[1])
            area = (x_max  - x_min) * (y_max - y_min)
            labels.append((frame, obj_type, house_number, street_name, False, x_min, x_max, y_min, y_max, area))
    label_df = pd.DataFrame(labels, columns = ["frame", "obj_type", "house_number", "street_name", "is_goal", "x_min", "x_max", "y_min", "y_max", "area"])
    print("num labels failed to parse: " + str(len(failed_to_parse)))

    # find target panos -- they are the ones with the biggest bounding box an the house number
    doors = label_df[label_df.obj_type == "door"]
    addresses = set([x.house_number + "-" + x.street_name for i, x in doors[["house_number", "street_name"]].iterrows()])

    for address in addresses:
        house_number, street_name = address.split("-")
        matched_doors = label_df[(label_df.obj_type == "door") & (label_df.house_number == house_number) & (label_df.street_name == street_name)]
        label_df.at[matched_doors.area.idxmax(), "is_goal"] = True
    return label_df

def construct_spatial_graph(coords_df, is_mini_hyrule, do_plot):
    """ Filter the pano coordinates by spatial relation and write the filtered graph to disk"""

    coords_df, node_blacklist, edge_blacklist, add_edges = cleanup_graph(coords_df, is_mini_hyrule)
    coords_df = coords_df[~coords_df.index.isin(node_blacklist)]

    # Init graph
    G = nx.Graph()
    G.add_nodes_from(coords_df.index)

    nodes = G.nodes
    for node_1_idx in tqdm(nodes, desc="Adding edges to graph"):
        meta = coords_df[coords_df.index == node_1_idx]
        coords = np.array([meta['x'].values[0], meta['y'].values[0], meta['z'].values[0]])
        G.nodes[node_1_idx]['coords'] = coords
        G.nodes[node_1_idx]['timestamp'] = meta.timestamp
        G.nodes[node_1_idx]['angle'] = meta.angle
        G.nodes[node_1_idx]['old_index'] = node_1_idx

        radius = 1.1
        nearby_nodes = coords_df[(coords_df.x > coords[0] - radius) & (coords_df.x < coords[0] + radius) & (coords_df.y > coords[1] - radius) & (coords_df.y < coords[1] + radius)]
        for node_2_idx in nearby_nodes.index:
            if node_1_idx == node_2_idx:
                continue
            meta2 = coords_df[coords_df.index == node_2_idx]
            coords2 = np.array([meta2['x'].values[0], meta2['y'].values[0], meta2['z'].values[0]])
            G.nodes[node_2_idx]['coords'] = coords2
            node_distance = np.linalg.norm(coords - coords2)
            G.add_edge(node_1_idx, node_2_idx, weight=node_distance)

    for n1, n2 in edge_blacklist:
        if n1 in G.nodes and n2 in G.nodes:
            G.remove_edge(n1, n2)

    for n1, n2 in add_edges:
        if n1 in G.nodes and n2 in G.nodes:
            G.add_edge(n1, n2)

    G, coords_df = clean_positions(G, coords_df)

    if do_plot:
        pos = {k: v.get("coords")[0:2] for k, v in G.nodes(data=True)}
        nx.draw_networkx(G, pos,
                         nodelist=G.nodes,
                         node_color='r',
                         node_size=10,
                         alpha=0.8,
                         with_label=True)
        #nx.draw(G, pos,node_color='r', node_size=1)

        plt.axis('equal')
        plt.show()

    mapping = coords_df.loc[coords_df.index, 'frame'].to_dict()
    G = nx.relabel_nodes(G, mapping)

    return G, coords_df

def process_images(paths):
    thumbnails = np.zeros((len(paths), 84, 224, 3))
    frames = np.zeros(len(paths))

    # Get panos and crop'em into thumbnails
    for idx, path in enumerate(tqdm(paths, desc="Loading thumbnails")):
        frame = int(path.split("_")[-1].split(".")[0])
        frames[idx] = frame
        image = cv2.imread(path)
        image = cv2.resize(image, (width, height))[:, :, ::-1]
        image = image[crop_margin:height - crop_margin]
        thumbnails[idx] = image

    images = {frame: img for frame, img in zip(frames, thumbnails)}
    return images

def cleanup_graph(coords_df, is_mini_hyrule):
    node_blacklist = [0, 928, 929, 930, 931, 1138, 6038, 6039, 5721, 5722, 6091, 6090, \
                      6082, 6197, 6088, 4809, 5964, 5504, 5505, 5467, 5514, 174,    \
                      188, 189, 190, 2390, 2391, 2392, 2393, 1862, 1863, 1512, 1821, 4227,\
                      1874, 3894, 3895, 3896, 3897, 3898, 2887, 608, 3025, 3090, 3013, 3090, \
                      780, 781, 162, 1822, 1725, 1726, 1513, 3875, 4842, 4870, 4907, 4717, \
                      6214, 6215, 5965, 5966, 4715, 3362, 5358, 3419, 5457, 5458, 3418]
    node_blacklist.extend([x for x in range(1330, 1346)])
    node_blacklist.extend([x for x in range(3034, 3071)])
    node_blacklist.extend([x for x in range(879, 892)])
    node_blacklist.extend([x for x in range(2971, 2983)])
    node_blacklist.extend([x for x in range(2888, 2948)])
    node_blacklist.extend([x for x in range(2608, 2629)])
    node_blacklist.extend([x for x in range(3091, 3098)])
    node_blacklist.extend([x for x in range(704, 780)])
    node_blacklist.extend([x for x in range(118,128)])
    node_blacklist.extend([x for x in range(5724, 5748)])
    node_blacklist.extend([x for x in range(6186, 6197)])
    node_blacklist.extend([x for x in range(4891, 4896)])
    node_blacklist.extend([x for x in range(6083, 6088)])
    node_blacklist.extend([x for x in range(5516, 5594)])
    node_blacklist.extend([x for x in range(5955, 5964)])
    node_blacklist.extend([x for x in range(5459, 5467)])
    node_blacklist.extend([x for x in range(3400, 3418)])
    node_blacklist.extend([x for x in range(4261, 4266)])
    node_blacklist.extend([x for x in range(5506, 5514)])
    node_blacklist.extend([x for x in range(3876, 3894)])
    node_blacklist.extend([x for x in range(5340, 5358)])
    node_blacklist.extend([x for x in range(1122, 1132)])
    node_blacklist.extend([x for x in range(652, 704)])
    node_blacklist.extend([x for x in range(3899, 4227)])
    node_blacklist.extend([x for x in range(2394, 2410)])
    node_blacklist.extend([x for x in range(585, 608)])
    node_blacklist.extend([x for x in range(2043, 2057)])
    node_blacklist.extend([x for x in range(2244, 2252)])
    node_blacklist.extend([x for x in range(2847, 2887)])
    node_blacklist.extend([x for x in range(3636, 3652)])
    node_blacklist.extend([x for x in range(2834, 2847)])
    node_blacklist.extend([x for x in range(3652, 3661)])
    node_blacklist.extend([x for x in range(4228, 4261)])
    node_blacklist.extend([x for x in range(3251, 3257)])
    node_blacklist.extend([x for x in range(3229, 3237)])
    node_blacklist.extend([x for x in range(1835, 1843)])
    node_blacklist.extend([x for x in range(5102, 5113)])
    node_blacklist.extend([x for x in range(2828, 2834)])
    node_blacklist.extend([x for x in range(2605, 2608)])
    node_blacklist.extend([x for x in range(3632, 3636)])
    node_blacklist.extend([x for x in range(3661, 3667)])
    node_blacklist.extend([x for x in range(4712, 4715)])
    node_blacklist.extend([x for x in range(5286, 5338)])
    node_blacklist.extend([x for x in range(6216, 6247)])
    node_blacklist.extend([x for x in range(4852, 4857)])
    node_blacklist.extend([x for x in range(4874, 4884)])
    node_blacklist.extend([x for x in range(6092, 6096)])
    node_blacklist.extend([x for x in range(5748, 5754)])
    node_blacklist.extend([x for x in range(5716, 5724)])

    edge_blacklist = [(913, 915), (835, 925), (824, 826), (835, 837), (900, 902), (901, 903),        \
                      (1534, 1536), (1511, 1724), (191, 137), (50, 172), (612, 782), (1135, 1714),   \
                      (109, 107), (1875, 1824), (1771, 1875), (4843, 4845), (4849, 4847),            \
                      (4860, 4862), (4861, 4863), (4909, 4813), (4910, 4908), (4506, 4897),          \
                      (4502, 4897), (5754, 5756), (5758, 5756)]
    add_edges = [(902,1329), (893, 894), (894, 895), (651, 2970), (638, 2983), (637, 2983), 		 \
                 (2637, 3098), (2629, 3088), (2954, 3026), (3077, 3078), (3109, 3110), (2948, 2607), \
                 (0, 1722), (6161, 6162), (6147, 6146), (6172, 6173), (6174, 6175), (4465, 4906),    \
                 (6212, 6213), (4465, 4464), (4467, 4466), (6037, 5594), (5458, 3418), (5129, 5128), \
                 (100, 101), (98, 1348), (99, 1348), (3630, 3631), (3631, 3632), (3632, 3633),       \
                 (3634, 3635), (2375, 3661), (1834, 2234), (1834, 2233), (3366, 3367), (2827, 2363), \
                 (379, 611), (2948, 3110), (2604, 3110), (3076, 3077), (782, 379), (782, 380),       \
                 (380, 611), (1, 1722), (1722, 185), (1722, 1121), (107, 1712), (2042, 51),          \
                 (173, 2042), (2375, 3667), (2374, 3667), (2364, 2827), (163, 2349), (3631, 2389),   \
                 (3631, 2304), (1727, 1513), (1727, 1724), (1514, 1724), (1514, 1727), (1781, 1853), \
                 (1781, 3228), (2242, 3116), (3387, 5113), (3377, 3874), (3177, 5358), (5284, 5285), \
                 (5285, 4840), (5285, 4841), (4841, 4843), (4851, 4857), (4842, 4843), (4845, 4846), \
                 (4852, 4853), (4872, 4873), (4874, 4875), (4876, 5308), (4873, 4776), (4873, 4775), \
                 (4811, 4764), (4812, 4764), (4718, 4463), (6213, 4504), (6213, 4503), (6213, 4505), \
                 (6096, 4489), (6096, 4896), (5754, 6171), (5754, 6170), (6170, 6171), (6179, 6059), \
                 (5715, 6040), (6178, 6177), (6045, 6210), (6177, 6210), (6177, 6044), (5954, 5480), \
                 (5359, 3176), (5359, 4711), (5359, 3177), (5456, 3188), (5456, 3187), (3186, 3248), \
                 (5456, 3186), (161, 163)]

    if is_mini_hyrule:
        node_blacklist.extend([x for x in range(877, 879)])
        node_blacklist.extend([x for x in range(52, 56)])
        node_blacklist.extend([x for x in range(31, 39)])
        node_blacklist.extend([x for x in range(2040, 2045)])
        node_blacklist.extend([x for x in range(2057, 2063)])
        node_blacklist.extend([x for x in range(3661, 3669)])
        node_blacklist.extend([x for x in range(780, 784)])
        box = (24, 76, -125, 10)
        coords_df = coords_df[((coords_df.x > box[0]) & (coords_df.x < box[1]) & (coords_df.y > box[2]) & (coords_df.y < box[3]))]
    return coords_df, node_blacklist, edge_blacklist, add_edges

def clean_positions(G, coords_df):
    nodes_to_move = [([n for n in range(5129, 5266)], [x*0.085 for x in range(0, 266-129)], [y*0.07 for y in range(0, 266-129)]), \
                     ([n for n in range(5266, 5285)], [x*0.04 + 11.6 for x in range(0, 266-170)], [y*0.1 + 9.5 for y in range(0, 266-170)]), \
                     ([5285], [11.3], [8.8]),                                                                                                \
                     ([n for n in range(4843, 4852)], [-0.2, -0.2, -0.2, -0.3, 0, 0.2, 0.2, 0.3, 0], [-1, 0, 1, 1, 1.3, 1.8, 2.2, 3.2, 3.8]),\
                     ([n for n in range(4857, 4874)], [-x*0.7 for x in range(0, 74-57)], [y*0.2 for y in range(0, 74-57)]),                  \
                     ([4872, 4873, 4762, 4464, 4718, 6172, 6171, 6170], [-0.2, -0.7, 0, 0,-0.4, 0.6, 0.6, 0], [0.3, 0, -0.3, 0.2, 0, 0, 0, -0.5]),\
                     ([n for n in range(6096, 6142)], [(46-x)*0.05 for x in range(0, 46)], [(46-y)*0.01 for y in range(0, 46)]),             \
                     ([n for n in range(5754, 5784)], [-(30-x)*0.02 for x in range(0, 30)], [(30-y)*0.07 for y in range(0, 46)]),            \
                     ([n for n in range(5700, 5716)], [x*0.02 for x in range(0, 16)], [y*0.15 for y in range(0, 16)]),                       \
                     ([n for n in range(5594, 5601)], [(7-x)*0.05 for x in range(0, 7)], [0 for y in range(0, 7)]),                          \
                     ([n for n in range(5920, 5955)], [-x*0.02 for x in range(0, 35)], [0 for y in range(0, 35)]),                           \
                     ([3305, 609, 610, 128, 129, 5515], [0, -0.3, -0.2, 0, 0, 1], [-0.2, 0, 0, -0.4, -0.3, 0.3]),                            \
                     ([n for n in range(5359, 5371)], [(12-x)*0.1 for x in range(0, 12)], [(12-y)*0.02 for y in range(0, 12)]),              \
                     ([n for n in range(5435, 5457)], [x*0.1 for x in range(0, 22)], [0 for y in range(0, 22)])]
    for nodes, x, y in nodes_to_move:
        for idx, node in enumerate(nodes):
            try:
                G.nodes[node]['coords'][0] += x[idx]
                G.nodes[node]['coords'][1] += y[idx]
                coords_df.at[node, 'x'] = coords_df.loc[node]['x'] + x[idx]
                coords_df.at[node, 'y'] = coords_df.loc[node]['y'] + y[idx]
            except:
                pass

    return G, coords_df

def label_segments(coords_df):
    coords_df["type"] = None
    coords_df["group"] = None
    street_segments = [(25.3, 30.5, -113, -1.8), (33, 38, -113.25, -2.5), (62, 66, -115.8, -3.39), (68.7, 76, -116, -4), (35, 63.2, -125, -121), (37, 62.5, -116.8, -115.1), (38.3, 62.2, -3.3, -0.8), (35.75, 62.5, 4.8, 6.6), (1.8, 30.2, 105, 112.3)]
    intersections = [(24.7, 35, -123.5, -112.7), (61.6, 73.6, -124.5, -116), (26.38, 38.23, -2.4, 6.8), (62.0, 72.2, -3.5, 6.2)]
    street_segments.extend([(-2.9, 25.4, -122.5, -121), (-2.7, 24.7, -114.9, -113.5), (0.9, 28.4, -2.0, 0.88), (0.59, 26.6, 5.1, 8.3), (-11.5, -7, -113.5, -1), (-3.8, 1.3, -113.8, -0.4), (26.5, 30.6, -112.5, -1.9)])
    intersections.extend([(-11.5, -2.7, -123, -113.5), (-7.6, 0.52, -1.25, 8.5)])
    street_segments.extend([(-7.7, -4.8, 7.7, 105), (-1, 1.6, 7.9, 105.1), (25.5, 31.2, 6.9, 105.1), (33, 39.9, 6.6, 105.2), (61.9, 69.6, 6.4, 137.6), (71.5, 76, 4.8, 137.4), (30.9, 33.4, 111.6, 135.4), (38.4, 40.3, 111.4, 135), (3.5, 30.9, 134.5, 137.7), (3.2, 31.5, 142, 143.6), (40.9, 67.6, 135.5, 138.6), (40.8, 67, 142.7, 145.6)])
    intersections.extend([(-8.7, 1.3, 105.2, 112.1), (30.1, 39.8, 105, 111.8), (-5, 4, 134.8, 143.3), (30.9, 41.3, 134.5, 144.6), (67.2, 77.9, 137, 145.5)])
    street_segments.extend([(-6.5, -3.3, 142.8, 260.6), (0, 3, 143, 260), (31, 35.1, 32.1, 260.6), (38.7, 43, 144.2, 261), (67, 70.6, 145.3, 260.6), (72, 89.5, 145.3, 258.9), (2.5, 33.7, 260.4, 262.3), (2.75, 33.9, 270.4, 272.1), (42.7, 67.9, 260.6, 262.2), (42.9, 67.2, 270, 273)])
    intersections.extend([(-7.7, 2.6, 260.2, 271.6), (33.6, 42.75, 260.69, 272.9), (67.5, 92.7, 258.9, 274)])
    for idx, box in enumerate(street_segments):
        segment = coords_df[((coords_df.x > box[0]) & (coords_df.x < box[1]) & (coords_df.y > box[2]) & (coords_df.y < box[3]))]
        segment["type"] = "street_segment"
        segment["group"] = idx
        coords_df.loc[segment.index] = segment

    for idx, box in enumerate(intersections):
        intersection = coords_df[((coords_df.x > box[0]) & (coords_df.x < box[1]) & (coords_df.y > box[2]) & (coords_df.y < box[3]))]
        intersection["type"] = "intersection"
        intersection["group"] = idx
        coords_df.loc[intersection.index] = intersection
    return coords_df

def create_dataset(data_path="/data/hyrule/", do_images=True, do_graph=True, do_plot=False, limit=None, is_mini_hyrule=False):
    """
    Loads in the pano images from disk, crops them, resizes them, and writes them to disk.
    Then pre-processes the pose data associated with the image and calls the fn to create the graph and to process the labels
    """
    data_path = os.getcwd() + data_path

    if do_graph:
        if limit:
            coords = np.load(data_path + "processed/pos_ang.npy")[:limit]
        else:
            coords = np.load(data_path + "processed/pos_ang.npy")
        coords_df = pd.DataFrame({"x": coords[:, 2], "y": coords[:, 3], "z": coords[:, 4], "angle": coords[:, -1], "timestamp": coords[:, 1], "frame": [int(x) for x in coords[:, 1]*30]})
        G, coords_df = construct_spatial_graph(coords_df, is_mini_hyrule, do_plot)
        nx.write_gpickle(G, data_path + "processed/graph.pkl")
        label_paths = [data_path + "raw/labels/pano_" + str(frame).zfill(6) + ".xml" for frame in coords_df["frame"].tolist()]
        label_paths = [path for path in label_paths if os.path.isfile(path)]
        label_df = process_labels(label_paths)

        meta_df = label_df.merge(coords_df, how="outer", on="frame")
        meta_df = label_segments(meta_df)
        label_index = meta_df.groupby(meta_df.frame).cumcount()
        meta_df.index = pd.MultiIndex.from_arrays([meta_df.frame, label_index], names=["frame", "label"])
        meta_df.sort_index(inplace=True)
        meta_df.to_hdf(data_path + "processed/meta.hdf5", key="df", index=False)
    else:
        meta_df = pd.read_hdf(data_path + "processed/meta.hdf5", key="df", index=False)
        G = nx.read_gpickle(data_path + "processed/graph.pkl")

    # Get png names and apply limit
    img_paths = [data_path + "panos/pano_" + str(frame).zfill(6) + ".png" for frame in coords_df["frame"].tolist()[1:]]
    if limit:
        img_paths = img_paths[:limit]

    if do_images:
        images = process_images(img_paths)
        f = gzip.GzipFile(data_path + "processed/images.pkl.gz", "w")
        pickle.dump(images, f)
        f.close()

create_dataset(data_path="/data/hyrule/", do_images=False, do_graph=True, do_plot=True, is_mini_hyrule=False)
