import pandas as pd
try:
    import networkx as nx
except ImportError:
    nx = None


def dates_to_timestamps(s):
    return (pd.to_datetime(s) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')


def pcore_filter(data, pcore, userid, itemid, keep_columns=True):
    if nx is None:
        pcore_data = pcore_filter_iter(data, pcore, userid, itemid)
    else:
        pcore_data = pcore_filter_graph(data, pcore, userid, itemid)
    if keep_columns and (data.shape[1] > 2):
        remaining_data = data.loc[pcore_data.index, data.columns.drop(pcore_data.columns)]
        pcore_data = pd.concat([pcore_data, remaining_data], axis=1)    
    return pcore_data


def pcore_filter_iter(data, pcore, userid, itemid):
    while pcore: # do only if pcore is specified
        item_check = True
        valid_items = data[itemid].value_counts() >= pcore
        if not valid_items.all():
            valid_items_idx = valid_items.index[valid_items]
            data = data.loc[lambda x: x[itemid].isin(valid_items_idx)]
            item_check = False
            
        user_check = True
        valid_users = data[userid].value_counts() >= pcore
        if not valid_users.all():
            valid_users_idx = valid_users.index[valid_users]
            data = data.loc[lambda x: x[userid].isin(valid_users_idx)]
            user_check = False
        
        if user_check and item_check:
            break
    return data.copy()


def pcore_filter_graph(data, pcore, userid, itemid):
    g, node_prefix = bipartite_graph_from_df(data, userid, itemid)
    g_pcore = nx.k_core(g, k=pcore) # apply p-core filtering
    pcore_data = pd.DataFrame.from_records(
        read_bipartite_edges(g_pcore, part=0), # iterate user-wise
        columns=[userid, itemid, 'index']
    ).set_index('index')
    # remove user/item node identifiers and restore source dtypes
    for field, prefix in node_prefix.items():
        start = len(prefix)
        pcore_data.loc[:, field] = pcore_data[field].str[start:].astype(data.dtypes[field])
    return pcore_data

def bipartite_graph_from_df(df, top, bottom):
    '''
    Construct bipartite top-bottom graph from pandas DataFrame.
    Assumes DataFrame has `top` and `bottom` columns.
    Edge weights are used to store source DataFrame index.
    '''
    node_prefix = {top: 't-', bottom: 'b-'}
    nx_data = (
        df[[top, bottom]]
        .agg({ # add node identifiers for bipartite graph
            top: lambda x: f"{node_prefix[top]}{x}",
            bottom: lambda x: f"{node_prefix[bottom]}{x}",
        })
    )
    g = nx.Graph()
    g.add_nodes_from(nx_data[top].unique(), bipartite=0)
    g.add_nodes_from(nx_data[bottom].unique(), bipartite=1)
    edge_iter = (
        nx_data
        .reset_index()
        [[top, bottom, 'index']] # keep source index
        .itertuples(index=False, name=None)
    )
    g.add_weighted_edges_from(edge_iter)
    return g, node_prefix

def read_bipartite_edges(graph, part=0):
    weighted = nx.is_weighted(graph)
    nodes = (node for node, prop in graph.nodes.items() if prop["bipartite"]==part)
    for node in nodes:
        yield from graph.edges(node, data='weight' if weighted else False)