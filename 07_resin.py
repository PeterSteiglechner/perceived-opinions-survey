#%%
print("ResIN")
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from consts import *
import numpy as np
from itertools import combinations
import networkx as nx
import scipy.stats as stt

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from matplotlib.font_manager import FontProperties
def text_marker(s, fontsize=10, font=None):
    fp = FontProperties(family=font) if font else None
    tp = TextPath((0,0), s, size=fontsize, prop=fp)
    bbox = tp.get_extents()
    cx, cy = (bbox.x0+bbox.x1)/2, (bbox.y0+bbox.y1)/2
    return Affine2D().translate(-cx,-cy).transform_path(tp)



plt.rcParams.update({"font.size":9})
plt.rcParams.update({"figure.figsize":(16/2.54, 9/2.54)})
sns.set_style("ticks")
sns.set_context("paper")

#%%
df_p = pd.read_csv("processed_data/2026-06-19_data_processed_participant_withAllIssueWeights_justParties.csv")
df_diff = pd.read_csv("processed_data/2026-06-19_data_processed_differences_withAllIssueWeights_justParties.csv")
#%%

# i, w = df_p[["id", "wave"]].sample()
# row = f"id=={i} and wave=={w}"
row = f"wave==1"
list_of_node_variables = [f"x_self_{q}" for q in questions_sc]
heat_variable = ["lr"]

labels_likert={
    2:[r"$-$",r"$+$"],
    3:[r"$-$", r"$0$", r"$+$"],
    5:[r"$--$",r"$-$", r"$0$", r"$+$", r"$++$"],
    7:[r"$---$", r"$--$",r"$-$", r"$0$", r"$+$", r"$++$", r"$+++$"],
}
def get_dummy(row , colnames= list_of_node_variables, n_likert = 3):
    if type(row)== str:
        d = df_p.query(row)
    else: 
        d = row
    df_nodes = d[colnames]
    df_heat = d[heat_variable]

    cuts = np.linspace(-1.01,1.01, n_likert+1)
    for c in colnames:
        df_nodes[c] = pd.cut(df_nodes[c], bins=cuts, labels=labels_likert[n_likert])
    
    # TAKEN FROM DINO
    df_dummy = pd.DataFrame()
    df = df_nodes.copy() # Select the dataframe of the nodes
    df = df.rename(columns=dict(zip(colnames, qs)))

    for col in qs: # For each column...
        values = (df[col].unique()) # ... get the list of the possible responses (i.e. nodes)
        for value in values: # For each response
            if type(value) == str: # check if the answer is type string
                name = str(col)+":"+str(value) # get the names as col:response
                df_dummy[name] = df[col] == value # get dummy-coded column

            else:
                if np.isnan(value): # if it's a NAused answer
                    name = str(col)+":"+"NA" 
                    df_dummy[name] = np.isnan(df[col])
                else: # Otherwise
                    name = str(col)+":"+str(value) 
                    df_dummy[name] = df[col] == value
        # if not  str(col)+":"+"NA" in  df_dummy.columns:
        #     df_dummy[ str(col)+":"+"NA" ] = False
    for col in qs:
        for value in labels_likert[n_likert]: # For each response
            if not str(col)+":"+str(value) in df_dummy.columns:
                df_dummy[str(col)+":"+str(value)] = False
    return df_dummy
# %%
def phi_(n11,n00,n10,n01):
    n1p = n11+n10
    n0p = n01+n00
    np1 = n01+n11
    np0 = n10+n00
    
    num = n11*n00-n10*n01
    den_ = n1p*n0p*np0*np1
    
    if den_==0:
        phi_=np.nan
    else:
        phi_ = num/np.sqrt(den_)
    return phi_

def p_val(r,L):
    den = np.sqrt(1-r**2)
    deg_free = L-2
    if den==0:
        p = 0
    else:
        num = r*np.sqrt(deg_free)
        t = num/den
        p = stt.t.sf(abs(t), df=deg_free)*2
    return p

def phi(x,y,get_p=False):
    
    m_eq = x==y
    m_diff = np.logical_not(m_eq)
    
    n11 = float(np.sum(x[m_eq]==True))
    n00 = float(np.sum(x[m_eq]==False))
    
    n10 = float(np.sum(x[m_diff]==True))
    n01 = float(np.sum(y[m_diff]==True))
    
    phi_val = phi_(n11,n00,n10,n01)
    
    if get_p:
        p = p_val(phi_val,len(x))
        return phi_val, p
    else:
        return phi_val

def make_graph_(df, list_of_nodes, alpha=0.05, get_p=True, remove_nan=False, remove_non_significant=False, exclude_same_question=True):
    
    if get_p==False and remove_non_significant==True:
        print("Warning: Setting remove_non_significant to False as get_p is False!")
        remove_non_significant=False
    
    G = nx.Graph()
    G.add_nodes_from(list_of_nodes)

    for i, node_i in enumerate(list_of_nodes):
        for j, node_j in enumerate(list_of_nodes):
            
            if j <= i: # do not run the same couple twice
                continue
                            
            basename1 = node_i.split(sep=':')[0]
            basename2 = node_j.split(sep=':')[0]
            
            if exclude_same_question:
                if basename1 == basename2: # if they belong to the same item
                    continue

            # Get the two columns
            c1 = df[node_i]
            c2 = df[node_j]
            
            if remove_nan:
                if ("NA" in node_i) or ("NA" in node_j):
                    continue
                
                c1_n = df[basename1+":NA"] # get the NAused values of each item
                c2_n = df[basename2+":NA"]
                
                mask = np.logical_not(np.logical_or(c1_n, c2_n)) # get a mask of the NAused values
                
                c1 = c1[mask] # select only the non-nan element
                c2 = c2[mask]
            
            if get_p:
                (r,p) = phi(c1,c2, get_p=True)
            else:
                r = phi(c1,c2, get_p=False)
            
            # Check if there are the conditions for drawing a node
            if remove_non_significant: 
                condition = r>0 and p<alpha
            else:
                condition = r>0

            if condition:
                G.add_weighted_edges_from([(node_i,node_j,r)],weight='weight')
                if get_p:
                    G.add_weighted_edges_from([(node_i,node_j,p)],weight='p')
                    sig = float(p<alpha) # Boolean are not accepted as edge weight
                    G.add_weighted_edges_from([(node_i,node_j,sig)],weight='sig')
    return G


#%%
def do_PCA(G):
    isolated = list(nx.isolates(G))
    connected = G.copy()
    connected.remove_nodes_from(isolated)
    pos = nx.spring_layout(connected,iterations=5000) # Get the positions with the spring layout
    print(pos)
    for n in isolated:
        pos[n] = (0,0)

    # Restructure the data type
    pos2 = [[],[]]
    key_list = [] # ordered list of the nodes
    for key in pos:
        pos2[0].append(pos[key][0])
        pos2[1].append(pos[key][1])
        key_list.append(key)

    # Use PCA to rotate the network in such a way that the x-axis is the main one
    pos3 = []
    for key in pos:
        pos3.append([pos[key][0],pos[key][1]])

    pca = PCA(n_components=2)
    pca.fit(pos3)
    x_pca = pca.transform(pos3)
    return x_pca

# Get the graph


def plot_resin(df_dummy, pos=None, ax=None, fig=None, n_likert = 3):
    G = make_graph_(df=df_dummy, list_of_nodes=df_dummy.columns, alpha=alpha, get_p=get_p,
                     remove_non_significant=remove_non_significant, remove_nan=remove_nan,
                     exclude_same_question=True)
    print("Complete!")

    if pos is None:
        # No reference given -> compute a fresh layout for this graph (e.g. for wave 1, or your master layout)
        x_pca = do_PCA(G)
        key_list = list(G.nodes())
        pos_dict = {key_list[i]: x_pca[i] for i in range(len(key_list))}
    else:
        # Reference positions given -> only keep nodes that exist in it, drop the rest
        key_list = [n for n in G.nodes() if n in pos]
        dropped = [n for n in G.nodes() if n not in pos]
        if dropped:
            print(f"Dropping {len(dropped)} nodes not in reference layout: {dropped}")
        G = G.subgraph(key_list).copy()
        pos_dict = {n: pos[n] for n in key_list}
        x_pca = np.array([pos_dict[n] for n in key_list])

    levels_present = [l for l in labels_likert[n_likert]+["NA"] if any(n.split(":")[1] == l for n in G.nodes())]
    

    # --- 1. Define marker shapes for the 3 Likert levels ---
    likert_shapes = {
            "$+$": text_marker(r"▲", font="DejaVu Sans"),
            "$0$":  text_marker(r"◯", font="DejaVu Sans"),
            "$-$": text_marker(r"▼", font="DejaVu Sans"),
        "NA": "*",       # star for missing/NAused, if present
    }
    fill_map = {r"$+$": True, r"$0$": False,  r"$-$": True, r"NA": True}

    if n_likert==5: 
        likert_shapes = {
            "$++$": text_marker(r"▲", font="DejaVu Sans"),
            "$+$":  text_marker(r"△", font="DejaVu Sans"),
            "$0$":  text_marker(r"◯", font="DejaVu Sans"),
            "$-$":  text_marker(r"▽", font="DejaVu Sans"),
            "$--$": text_marker(r"▼", font="DejaVu Sans"),
            "NA": text_marker(r"?"),
        }
        fill_map = {r"$++$": True, r"$+$": False, r"$0$": False, r"$-$": False, r"$--$": True, r"NA": True}
    if n_likert==7: 
        likert_shapes = {
            "$+++$": text_marker(r"▲", font="DejaVu Sans"),
            "$++$": text_marker(r"△", font="DejaVu Sans"),
            "$+$":  text_marker(r"◓", font="DejaVu Sans"),
            "$0$":  text_marker(r"◯", font="DejaVu Sans"),
            "$-$":  text_marker(r"◒", font="DejaVu Sans"),
            "$--$": text_marker(r"▽", font="DejaVu Sans"),
            "$---$": text_marker(r"▼", font="DejaVu Sans"),
            "NA": text_marker(r"?"),
        }
        fill_map = {r"$+++$": True, r"$++$": False, r"$+$": True, r"$0$": False, r"$-$": True,r"$--$": False, r"$---$": True, r"NA": True}

    # matplotlib doesn't have a literal "minus" marker, so X or a custom one reads cleanly;
    # if you want a true horizontal dash for "minus" use marker="_" instead of "X"

    # --- 2. Define a colormap for the 6 questions (basenames) ---

    basenames = sorted(set(n.split(":")[0] for n in G.nodes()))
    assert len(basenames) <= 6, f"Expected 6 questions, found {len(basenames)}: {basenames}"

    nameDict = {b:q for b, q in zip(basenames, qs)}
    cmap = plt.get_cmap("tab10")  # 6 well-separated, qualitative colors
    question_colors = {b: cmap(i) for i, b in enumerate(basenames)}

    # 3. Derive per-node color, shape, and count from node name "question:level"
    node_colors = []
    node_shapes = []
    node_basename = []
    node_counts = []

    for node in G.nodes():
        base, level = node.split(":")
        node_basename.append(base)
        node_colors.append(question_colors[base])
        node_shapes.append(likert_shapes.get(level, "*"))
        # df_dummy[node] is a boolean column -> count of True = number of observations at this node
        node_counts.append(int(df_dummy[node].sum()))

    node_counts = np.array(node_counts)
    print(node_counts)

    # Scale counts to marker sizes (area-based, so size differences read proportionally to sqrt(count))
    size_min, size_max = 10, 500  # tune to taste
    if node_counts.max() > node_counts.min():
        node_sizes = size_min + (size_max - size_min) * (
            np.sqrt(node_counts) - np.sqrt(node_counts.min())
        ) / (np.sqrt(node_counts.max()) - np.sqrt(node_counts.min()))
    else:
        node_sizes = np.full_like(node_counts, (size_min + size_max) / 2, dtype=float)

    size_map = {key: node_sizes[i] for i, key in enumerate(key_list)}
    count_map = {key: node_counts[i] for i, key in enumerate(key_list)}

    # --- 4. Layout: reuse your PCA-rotated spring layout ---
    pos_dict = {key_list[i]: x_pca[i] for i in range(len(key_list))}

    # --- 5. Edge weights ---
    edge_weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    if edge_weights.max() > edge_weights.min():
        lw = 2 + 8 * (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())
    else:
        lw = np.full_like(edge_weights, 1.0)

    edge_sig = np.array([G[u][v].get("sig", 1.0) for u, v in G.edges()])
    edge_alpha = np.where(edge_sig > 0, 0.7, 0.15)

    if ax is None:
        fig, ax = plt.subplots(figsize=(16/2.54, 11/2.54))
    
    # Draw edges (unchanged)
    for (u, v), w, a in zip(G.edges(), lw, edge_alpha):
        x = [pos_dict[u][0], pos_dict[v][0]]
        y = [pos_dict[u][1], pos_dict[v][1]]
        ax.plot(x, y, color="grey", linewidth=w, alpha=a, zorder=1)

    # Draw nodes, sized by count, grouped by (question, level)
    for base in basenames:
        for level in levels_present:
            idx = [i for i, n in enumerate(key_list) if n == f"{base}:{level}"]
            if not idx:
                continue
            xs = [x_pca[i, 0] for i in idx]
            ys = [x_pca[i, 1] for i in idx]
            ss = [size_map[key_list[i]] for i in idx]
            color = question_colors[base]
            fc = color if fill_map[level] else "none"

            ax.scatter(
                xs, ys,
                facecolors=fc,
                edgecolors=color,       # always colored, regardless of fill
                marker=likert_shapes[level],
                s=ss,
                linewidths=1.2,
                zorder=2,
            )

    # --- 7. Annotations ---
    for i, node in enumerate(key_list):
        base, level = node.split(":")
        qname = dict(zip(questions_sc, qs))
        ax.annotate(nameDict[base]+":"+level, (x_pca[i, 0], x_pca[i, 1]),
                    fontsize=10, ha="center", va="center", zorder=3,
                    xytext=(0, 20), textcoords="offset points")

    # --- 8. Legends: one for color (question), one for shape (likert direction) ---
    color_handles = [mpatches.Patch(color=question_colors[b], label=nameDict[b]) for b in basenames]
    shape_handles = [mlines.Line2D([], [], color="black", marker=likert_shapes[l], linestyle="None",
                                    markersize=9, label=l) for l in levels_present]

    legend1 = ax.legend(handles=color_handles, title="Question", loc="upper left",
                        bbox_to_anchor=(1.02, 1), frameon=False)
    ax.add_artist(legend1)
    ax.legend(handles=shape_handles, title="Position", loc="lower left",
            bbox_to_anchor=(1.02, 0), frameon=False)

    ax.axis("off")
    plt.tight_layout()
    return G, x_pca, fig, ax
# plt.savefig("resin_network_plot.png", dpi=300, bbox_inches="tight")
# plt.show()

# %%
# Parameters
remove_nan=False
get_p=True
remove_non_significant=False
alpha=0.05

n_likert = 5
df_dummy = get_dummy("wave==1", n_likert=n_likert)
G, x_pca, fig, ax = plot_resin(df_dummy, n_likert=n_likert)
key_list_w1 = list(G.nodes())
pos_dict_w1 = {key_list_w1[i]: x_pca[i] for i in range(len(key_list_w1))}

df_dummy = get_dummy("wave==2", n_likert=n_likert)
G, x_pca, fig, ax = plot_resin(df_dummy, n_likert=n_likert)
key_list_w2 = list(G.nodes())
pos_dict_w2 = {key_list_w1[i]: x_pca[i] for i in range(len(key_list_w2))}
pos_dict = {1:pos_dict_w1, 2:pos_dict_w2}

nx.to_pandas_edgelist(G)
# %%
w = 1
fig, axs = plt.subplots(2,4, sharex=True, sharey=True, figsize=(18,9))
for p, ax in zip(parties_full[:7], axs.flatten()):
    df_dummy = get_dummy(f"wave=={w} and party_close=='{p}'", n_likert=n_likert)
    G, pos, fig, ax = plot_resin(df_dummy, pos=pos_dict[w], ax=ax, fig=fig, n_likert=n_likert)
    ax.set_title(f"{p} (n={len(df_dummy)})")
fig.suptitle(f"ResIN of Own Opinions by party affiliation (wave {w})")
axs[-1,-1].axis("off")
plt.tight_layout()
plt.savefig(f"figs/resin_self_byParty_w{w}.png", dpi=600)

#%%

w = 1
fig, axs = plt.subplots(2,4, sharex=True, sharey=True, figsize=(18,9))
for p, ax in zip(parties_full[:7], axs.flatten()):
    df_dummy = get_dummy(f"wave=={w} and party_close=='{p}'", n_likert=n_likert)
    G, pos, fig, ax = plot_resin(df_dummy, pos=None, ax=ax, fig=fig, n_likert=n_likert)
    ax.set_title(f"{p} (n={len(df_dummy)})")
fig.suptitle(f"ResIN of Own Opinions by party affiliation (wave {w})")
axs[-1,-1].axis("off")
plt.tight_layout()
plt.savefig(f"figs/resin_self_byParty_w{w}_pcaPos.png", dpi=600)


# %%
w = 1
list_of_node_variables_diffD = [f"dot1_{q}" for q in questions_sc]
n_likert=3
fig, axs = plt.subplots(2,3, sharex=True, sharey=True, figsize=(18,9))
for p, ax in zip(["Left Party", "Green Party", "SPD", "CDU/CSU", "AfD"], axs.flatten()):
    id = df_p.query(f"wave=={w} and party_close=='{p}'")["id"].sample().iloc[0]
    print(p, id)
    df_dummy = get_dummy(df_diff.query(f"wave=={w} and id=={id} and (dot2=='AfD' and dot1 not in {partiesVars})"), colnames=list_of_node_variables_diffD, n_likert=n_likert)
    G, pos, fig2, ax = plot_resin(df_dummy, pos=pos_dict[w], ax=ax, n_likert=n_likert)
    ax.set_title(f"id={id} ({p})", fontsize=15)
fig.suptitle(f"ResIN of Social Circle for example individuals (wave {w})", fontsize=15)
axs[-1,-1].axis("off")
plt.tight_layout()
plt.savefig(f"figs/resin_socialCircle_byParty_w{w}.png", dpi=600)
# %%

# %%


#%%
w= 2
remove_non_significant = False
n_likert=3
df_dummy = get_dummy(f"wave=={w}", n_likert=n_likert)
G, x_pca, fig, ax = plot_resin(df_dummy, n_likert=n_likert)
key_list = [p for p in G.nodes()]
xx = x_pca[:,0]
yy = x_pca[:,1]

list_of_node_variables_diffD = [f"dot1_{q}" for q in questions_sc] 
df_diff_dummy_dot1 = get_dummy(df_diff.query(f"wave=={w}"), n_likert=n_likert, colnames=list_of_node_variables_diffD).loc[:, key_list]
df_diff_dummy_dot1["id"] = df_diff.query(f"wave=={w}")["id"]
df_diff_dummy_dot1["dot1"] = df_diff.query(f"wave=={w}")["dot1"]
df_diff_dummy_dot1["dot2"] = df_diff.query(f"wave=={w}")["dot2"]
list_of_node_variables_diffD = [f"dot2_{q}" for q in questions_sc] 
df_diff_dummy_dot2 = get_dummy(df_diff.query(f"wave=={w}"), n_likert=n_likert, colnames=list_of_node_variables_diffD).loc[:, key_list]
df_diff_dummy_dot2["dot1"] = df_diff.query(f"wave=={w}")["dot1"]
df_diff_dummy_dot2["dot2"] = df_diff.query(f"wave=={w}")["dot2"]
df_diff_dummy_dot2["id"] = df_diff.query(f"wave=={w}")["id"]



# %%
df_diff_dummy_dot1["x1"] = df_diff_dummy_dot1.drop(columns=["id", "dot1", "dot2"]).apply(lambda row: xx[row].mean(), axis=1)
df_diff_dummy_dot2["x2"] = df_diff_dummy_dot2.drop(columns=["id", "dot1", "dot2"]).apply(lambda row: xx[row].mean(), axis=1)
df_diff_dummy_dot1["y1"] = df_diff_dummy_dot1.drop(columns=["id", "dot1", "dot2", "x1"]).apply(lambda row: yy[row].mean(), axis=1)
df_diff_dummy_dot2["y2"] = df_diff_dummy_dot2.drop(columns=["id", "dot1", "dot2", "x2"]).apply(lambda row: yy[row].mean(), axis=1)

# %%
# df_diff_dummy_dot1.merge(df_diff.query(f"wave=={w}")[["id",  "pixel_dist", "sympathy", "pairwise_similarity"]], on=["id"],  )
# df_diff_dummy_dot2.merge(df_diff.query(f"wave=={w}")[["id",  "pixel_dist", "sympathy", "pairwise_similarity"]], on=["id"],  )

df_diff_x = df_diff.loc[df_diff.wave==w, ["id", "wave", "dot1", "dot2", "pixel_dist", "sympathy", "pairwise_similarity", "socialCloseness"] + [f"deltaX_{q}" for q in questions_sc]]
positions = df_diff_dummy_dot2[["id", "dot1", "dot2", "x2", "y2"]].merge(df_diff_dummy_dot1[["id", "dot1", "dot2", "x1", "y1"]], on=["id", "dot1", "dot2"], )
#%%
df_diff_x = df_diff_x.merge(positions[["id", "dot1" ,"dot2", "x1", "x2", "y1", "y2"]], on=["id", "dot1", "dot2"], )
# %%
df_diff_x["delta_resin_x"] = abs(df_diff_x["x2"] - df_diff_x["x1"])
df_diff_x["delta_resin_y"] = abs(df_diff_x["y2"] - df_diff_x["y1"])
df_diff_x["delta_resin"] = np.sqrt(df_diff_x["delta_resin_x"]**2 + df_diff_x["delta_resin_y"]**2)
deltacols = [f"deltaX_{q}" for q in questions_sc]
df_diff_x["meandeltaX_core"] = df_diff_x[[q for q in deltacols if not "internet" in q and not "east" in q ]].mean(axis=1)
df_diff_x["meandeltaX"] = df_diff_x[deltacols].mean(axis=1)
corr = df_diff_x[["delta_resin", "delta_resin_x", "delta_resin_y", "meandeltaX_core","meandeltaX"] + [f"deltaX_{q}" for q in questions_sc]+ ["pixel_dist", "sympathy", "pairwise_similarity","socialCloseness"]].corr()
corr = corr.rename(columns=dict(zip(deltacols, questions_sc)), index=dict(zip(deltacols, questions_sc)))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, mask=np.triu(np.ones(corr.shape), k=0), annot_kws={"fontsize":8})
# %%
