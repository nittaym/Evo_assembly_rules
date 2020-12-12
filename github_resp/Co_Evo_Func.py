'''
Created 06-2020

@author:Nittaym
'''
import numpy as np
import pandas as pd
import scipy.stats
import math
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
import itertools
import matplotlib as plt




def add_pseudocounts(count_table, sample_col='sample', add = 1):
    '''Add pseudocounts for all species that should be in a specific community.
-----------------------------------------------------------------------------------------
    Parameters:
    counts_table:
        A pd dataframe with rows ad datapoints and columns as species
    sample_col:
        the column in the data frame that indicates which species should be in the sample
        a str in the form sp1_sp2
-----------------------------------------------------------------------------------------
    Return:
        pd.Dataframe with added pseodocounts'''

    count_table[count_table[sample_col].split('_')] = count_table[count_table[sample_col].split('_')] + add
    return count_table


def Shannon(prob, neff=True, base =2):
    '''Calcuate the Shannon entropy for an array of probabilities.
-----------------------------------------------------------------------------------------
    Parameters:
    prob:
        np.array of probabilies where sum(prob) = 1
    neff:
        if True, return the the number equivalent
    base:
        the log base
-----------------------------------------------------------------------------------------
    Return:
        float, the Shannon entropy
        '''

    prob = prob[prob!=0] # remove zeroes
    sh = np.sum(prob*-1*(np.log(list(prob)/np.log(base))))
    if neff == True:
        sh = base**sh
    return sh

def euc_change(d, n = 1):
    '''Calculate the euclidean distance from each timepoint to the last measurment.
-----------------------------------------------------------------------------------------
    Parameters:
    d:
        dataframe with species as columns
-----------------------------------------------------------------------------------------
    Return:
        a pd.Dataframe, in every timepoint a float indicating the distance from last measurment.
        first timepoint would return NaN'''

    cs = pd.DataFrame(index = d.index, columns=['c'])
    for i in d.index:
        if (d.shift().loc[i, :].isnull().any() | d.loc[i, :].isnull().any()) == False:
            cs.loc[i, 'c'] = euclidean(d.loc[i, :].values, d.shift().loc[i, :].values)
    return cs/np.sqrt(n)

def dist_from_med(d, n=1):
    '''Compute the mean distacne from a medioid.
-----------------------------------------------------------------------------------------
    Parameters:
    d:
        dataframe or np.matrix with rows as samples and columns as observations
    '''

    dm = distance_matrix(d, d)
    minimum = np.argmin(dm.sum(axis = 0))
    return dm[minimum][np.arange(0, len(dm))!=minimum].mean()/np.sqrt(n)


def get_species_fraction(df, species, ident, transfer):
    '''Return the the fractions of a specific ident (plate+well) at a certain transfer.
-----------------------------------------------------------------------------------------
    Parameters:
    df:
        a dataframe with fractions, and ident column, and transfer coulmn
    species:
        the count coulmns
    ident:
        str, plate+well as p1A1
    transfer:
        float or int, the transfer to take fractions from
    '''

    frac = df[species][df['ident'] == ident][df['transfer'] == transfer].values
    return frac


def get_well_OD(df, ident, transfer):
    ''' Get the OD of a well (ident) at a speciefic transfer.
-----------------------------------------------------------------------------------------
    Parameters:
    df:
        OD data frame, with coulmn OD as values
    ident:
        str, plate+well as p1A1
    transfer:
        float or int, the transfer to take fractions from'''

    return df['OD'][df['ident'] == ident][df['transfer'] == transfer].values


def get_fractional_OD(fractions, ods, species, ident, transfer):
    '''Return the the fractional OD of a specific ident (plate+well) at certain transfer.
-----------------------------------------------------------------------------------------
    Parameters:
    fractions:
        a dataframe with fractions, and ident column, and transfer coulmn
    ods:
        OD data frame, with coulmn OD as values
    species:
        the count coulmns
    ident:
        str, plate+well as p1A1
    transfer:
        float or int, the transfer to take fractions from'''

    frac = get_species_fraction(fractions, species, ident, transfer)
    if len(frac) > 0:
        frac = frac[0]
    od = get_well_OD(ods, ident, transfer)
    return np.array([f * od for f in frac]).squeeze()


def randomize_incs(sample, incs, shape, bysp=False):
    ''' Shuffle the increasing values
-----------------------------------------------------------------------------------------
    Parameters:
    bysp:
        if bysp = False, shuffles the values across all species and communities
        if bysp = True, shuffles each species only with its own increasing values
    sample:
        str, the community, written as sp1_sp2
    incs:
        array/dict, values to choose randomly from. if bysp = True, should be dict with
        species names as keys.
    shape:
        number of species in the community
    '''

    if bysp == False:
        new = np.random.choice(incs, shape)
    else:
        new = [np.random.choice(incs[sp]) for sp in sample.split('_')]
    return new

def randomize_counts(count_table, sample_col='sample', dist='dirichlet', costume_dist=None):
    '''Return random frequencies of species ,{only for present speices}
-----------------------------------------------------------------------------------------
    Parameters:
    count_table:
        a table with species as columns, and a sample column that indicates which species should be there
        sample_col should be with str and as sp1_sp2
    dist:
        'costume'or 'dirichlet', if 'costume' choose randomly numbers from costume_dist. if dirichlet use draw from dirichlet
    costume_dist:
        array, costume distribution to draw numbers from
-----------------------------------------------------------------------------------------
    Return:
        a pd.Datafram, structured like count_table, but with random fractions.
    '''
    if dist == 'dirichlet':
        si = len(count_table[sample_col].split('_'))
        ran = scipy.stats.dirichlet(np.repeat(1, si)).rvs().squeeze()
        count_table[count_table[sample_col].split('_')] = ran
    elif dist == 'costume':
        si = len(count_table[sample_col].split('_'))
        ran = np.random.choice(costume_dist[costume_dist.apply(lambda x: len(x) == si)])
        count_table[count_table[sample_col].split('_')] = ran
    return count_table

# def randomize_counts(count_table, sample_col='sample'):
#     '''Return random (drawn from dirichlet) frequencies of species , but only for present speices
#     count_table: a table with species as columns, and a sample column that indicates which species should be there
#     sample_col should be with str and as sp1_sp2'''
#     si = len(count_table[sample_col].split('_')) #number of species in the commnity
#     ran = scipy.stats.dirichlet(repeat(1, si)).rvs().squeeze() #
#     count_table[count_table[sample_col].split('_')] = ran
#
#     return count_table


def calculate_nestedness(comm_matrix):
    ''''''
    comm_matrix = comm_matrix.reindex(comm_matrix.median(axis=1).sort_values(ascending=True).index,
                                      axis=0).reindex(comm_matrix.median(axis=1).sort_values(ascending=True).index,
                                                      axis=1)
    tr = triu(comm_matrix.values)
    return len(tr[tr < 0]) / (len(tr[tr < 0]) + len(tr[tr > 0]))


def most_frequent(List):
    ''''''
    List = list(List)
    return max(set(List), key=List.count)


def zscore(distance, mean_distance, sd_distance):
    ''''''
    return (distance - mean_distance) / sd_distance

def pmax(l):
    '''Return the frequency of the maximum element'''
    return max(l)/sum(l)


def summarize_repeatability(df):
    df['max'] = df['fold_increase'].apply(np.argmax)
    df = df.groupby(['sample', 'max'])['max'].count().groupby('sample').apply(pmax).reset_index()
    return df['max'].values



def get_competative_scores(sp, df, logit=False):
    '''Get the mean relative abundace over all communities a species was a part of
    sp: str, the query species
    df: Dataframe, with counts and a sample column as community
    logit: if true transform to logit'''

    in_community = lambda x: sp in x['sample'].split('_')
    other_species = lambda x: np.array(x.split('_'))[np.array(x.split('_')) != sp][0]
    if logit == True:
        result = df[sp][df.apply(in_community, axis=1)].apply(lambda x: x / (1 - x))
    else:
        result = df[sp][df.apply(in_community, axis=1)]
    result = pd.DataFrame(result)
    result['partner'] = df['sample'][df.apply(in_community, axis=1)].apply(other_species)
    result.index = df['ident'][df.apply(in_community, axis=1)].values

    return result


def get_improvment_values(sp, df, ti, tf, logit=True):
    '''Get the mean  increase in relative abundace over all communities a species was a part of
    sp: str, the query species
    df: Dataframe, with counts and a sample column as community
    ti: trasfer to start with
    tf: final t
    logit: if true transform to logit'''

    start = get_competative_scores(sp, df[df['transfer'] == ti], logit=False).groupby('partner').mean()
    end = get_competative_scores(sp, df[df['transfer'] == tf], logit=False).groupby('partner').mean()
    if logit == True:
        start = start / (1 - start)
        end = end / (1 - end)
    imp = pd.DataFrame(end[sp] / start[sp])
    #     imp['partner'] = imp['index'].apply(lambda x:df['sample'][df['ident']==x].values[0])
    #     imp['partner'] = imp['partner'].apply(lambda x:array(x.split('_'))[array(x.split('_'))!=sp][0])
    return imp

### Predictions




def return_pairwise(community):
    '''Return all the pairs composing a with the form sp1_sp2
    community:str, built as 'sp1_sp2_sp3'
    '''
    sps = community.split('_')
    comb = list(itertools.combinations(sps, 2))

    return [pair[0] + '_' + pair[1] for pair in comb]


def all_pairs_present(trio, pair_list):
    '''True if all pairwise pairs composing a trio are present in pair_list
    trio: str, built as 'sp1_sp2_sp3
    pair_list: list/np.array of strings with pairs as sp1_sp2'''

    return all([pr in pair_list for pr in return_pairwise(trio)])


def build_community_matrix(community, counts_table):
    '''Return a community matrix with columns coresponding
    to a species fraction when grown with a certein partner (row)'''

    sp_in = lambda x: sp in x['sample']
    others = lambda x: np.array(x['sample'].split('_'))[np.array(x['sample'].split('_')) != sp][0]
    mat = pd.DataFrame(columns=community.split('_'), index=community.split('_'))
    df = counts_table[counts_table['sample'].isin(return_pairwise(community))]
    if len(df) != 0:
        for sp in mat.columns:
            mat.loc[df[df.apply(sp_in, axis=1)].apply(others, axis=1).values, sp] = df[sp][
                df.apply(sp_in, axis=1)].values
    return mat


def pair_trio_prediction(trio, counts_table):
    '''Predict a trio composition from pairwise outcomes
    trio: str, in the form sp1_sp2_sp3
    counts_table: pd.Dataframe, with coloumns as observations and rows as sample
    return: a pd.Dataframe, with one row and columns as species
    return the predicted value which is calculated as the weighted geometric mean of the pairwise competitions
    '''

    sps = np.array(trio.split('_'))
    mat = build_community_matrix(trio, counts_table).transpose()
    outcome = pd.DataFrame(columns=sps)
    for sp in sps:
        f12 = mat.loc[sp, sps[sps != sp][0]]
        f13 = mat.loc[sp, sps[sps != sp][1]]
        w2 = np.sqrt(mat.loc[sps[sps != sp][0], sp] * mat.loc[sps[sps != sp][0], sps[sps != sp][1]])
        w3 = np.sqrt(mat.loc[sps[sps != sp][1], sp] * mat.loc[sps[sps != sp][1], sps[sps != sp][0]])
        outcome.loc[0, sp] = ((f12 ** w2) * (f13 ** w3)) ** (1 / (w2 + w3))
    outcome.loc[0, :] = outcome.loc[0, :] / outcome.sum(axis=1).values
    if all_pairs_present(trio, counts_table) == False:
        outcome == np.nan
    return outcome


def predict_by_grates(sample, species, g_rates, ks):
    sps = sample.split('_')
    rates = [ks[species == sp].values for sp in sps] * (
                1 - (np.log2(1500) / 48) / [g_rates[species == sp].values for sp in sps])
    ratio = rates / sum(rates)
    return ratio.squeeze()


def predict_max_inc(trio, pair_majorities, pair_means):

    '''Predict which species in a trio would increase by the biggest factor
    here we predict that if a species increased in the pairs it was in, it would increase in the trio
    if in each pair a different species increase, go to pred_no_h, in print no hierarchy
    -------------------------------------------------------------------------------------------------------
    pair_majorities:
        pd.Dataframe, with a sample column as pairs and a column indicating which and column most_frequent
        which indicates which species increased by the biggest factor (indicated as an index)
    pair_mean:
        pd.Dataframe, indicating the mean increase value of each oe of the species in each one of the pairs,
        used only if there is no hierarchy
    -------------------------------------------------------------------------------------------------------
    return
        str, species which is predicted to increase by the biggest factor in the trio'''
    temp = pair_majorities[pair_majorities['sample'].isin(return_pairwise(trio))]
    outcomes = [temp.loc[i, 'sample'].split('_')[temp.loc[i, 'most_freq']] for i in temp.index]
    if len(outcomes) == 3:
        out = most_frequent(outcomes)
        if len(set(outcomes)) == 3:
            out = pred_no_h(trio, pair_means)
            print(trio, ':no hierarchy')

    elif (len(outcomes) == 2) & (len(set(outcomes)) == 1):
        out = most_frequent(outcomes)
    else:
        out = np.nan

    return out

def pred_no_h(trio, pair_means):
    '''Predict which species in a trio would increase by the biggest factor
    here we predict that with the highest mean increase in pairs it would increase in the trio
    -------------------------------------------------------------------------------------------------------
    pair_mean:
        pd.Dataframe, indicating the mean increase value of each oe of the species in each one of the pairs,
        used only if there is no hierarchy
    -------------------------------------------------------------------------------------------------------
    return
        str, species which is predicted to increase by the biggest factor in the trio'''
    sp_in = lambda x:sp in x['sample'].split('_')
    where_in = lambda x: x['fold_increase'][np.where(np.array(x['sample'].split('_'))==sp)[0][0]]
    temp = pair_means[pair_means['sample'].isin(return_pairwise(trio))]
    avrs = []
    for sp in trio.split('_'):
        avrs.append(temp[temp.apply(sp_in, axis = 1)].apply(where_in, axis =1).mean())
    return trio.split('_')[np.where(np.array(avrs)==max(avrs))[0][0]]


def plot_trajectory(community, ax, cf, cm,
                    ides=None, ticksfont={},
                    labelfont={}, alpha='changing',
                    sps='com', xtick=[0, 200, 400]):
    d = cf[cf['sample'] == community]
    al = alpha
    if sps == 'com':
        sps = community.split('_')
    for i, ide in enumerate(set(d['ident'][:ides])):
        data = d[d['ident'] == ide][cf['total'] > 10]
        for st in sps:
            if alpha == 'changing':
                al = (i + 1) / len(set(d['ident'][:ides])) - 0.01
            colr = cm[st]
            ax.errorbar(y=data[st], x=data["Generation"], yerr=data['std'], fmt='|',
                        ecolor=colr, alpha=al)
            data.plot(y=st, x='Generation', ax=ax, legend=False,
                      colors=colr, alpha=al)
            data.plot.scatter(y=st, x='Generation', ax=ax, legend=False,
                              colors=colr, alpha=al)
    ax.set_ylim(0, 1.)
    ax.set_xticks(xtick);
    ax.set_xticklabels(xtick, fontdict=ticksfont)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels([0, 0.5, 1.], fontdict=ticksfont)
    ax.set_ylabel('Relative abundance', fontdict=labelfont)
    ax.set_xlabel('Generation', fontdict=labelfont)
    ax.set_xlim(0, 400)


def plot_trajectory2(community, ax, cf, cm,
                    ides=None, ticksfont={},
                    labelfont={}, alpha='changing',
                    sps='com', xtick=[0, 200, 400]):
    d = cf[cf['sample'] == community]
    al = alpha
    if sps == 'com':
        sps = community.split('_')
    for i, ide in enumerate(set(d['ident'][:ides])):
        data = d[d['ident'] == ide][cf['total'] > 10]
        for i, c in enumerate(['count1', 'count2']):
            if alpha == 'changing':
                al = (i + 1) / len(set(d['ident'][:ides])) - 0.01
            colr = cm[sps[i]]
            #             ax.errorbar(y = data[st], x= data["Generation"], yerr=data['std'],fmt='|',
            #                         ecolor= colr, alpha = al)
            data.plot(y=c, x='Generation', ax=ax, legend=False,
                      colors=colr, alpha=al)
            data.plot.scatter(y=c, x='Generation', ax=ax, legend=False,
                              colors=colr, alpha=al)
    ax.set_ylim(0, 1.)
    ax.set_xticks(xtick);
    ax.set_xticklabels(xtick, fontdict=ticksfont)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels([0, 0.5, 1.], fontdict=ticksfont)
    ax.set_ylabel('Relative \nabundance', fontdict=labelfont)
    ax.set_xlabel('Generation', fontdict=labelfont)
    ax.set_xlim(0, 60)
