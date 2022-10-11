import itertools
import collections

# Programming Assignment 1
# CS 315
# Written by Amethyst Skye

# txt_file: file which will be read and then processed as a list of lists
# return: list of lists, each basket is filled with items selected by a given customer
def read_input(txt_file):
	with open(txt_file, "r") as text_file:
		baskets_list = []
		for line in text_file.readlines():
			baskets_list.append(line.split())
	return baskets_list

# baskets_list: list of lists. Each list contains items in a given customers basket
# support: product pairs must appear together at least this many times
# return: dictionary containing frequent items
def find_frequent_items(baskets_list, support = 100):
    items = {}
    frequent_items = {}
    for basket in baskets_list:
        for item in basket:
            if item not in items:
                items[item] = 1
            else:
                items[item] += 1
    for key in items.keys():
        if items[key] >= support:
            frequent_items[key] = items[key]
    return(frequent_items)

# baskets_list: list of lists. Each list contains items in a given customers basket
# k: number of items in a group
# support: product groups must appear together at least this many times
# return: dictionary containing frequent item groups
def freq_k_items(baskets_list, k, support = 100):
    counter_dict = collections.Counter()
    for basket in baskets_list:
        basket = sorted(basket)
        combs_iterator = itertools.combinations(basket, k)
        for combs in combs_iterator:
            counter_dict[combs] += 1
    frequent_k_items = {i:counter_dict[i] for i in counter_dict if counter_dict[i] >= support}
    return frequent_k_items

# freq_items_dict: dictionary containing frequent items
# pairs_dict: dictionary containing frequent pairs
# return: list of sorted confidence levels for frequent pairs
def pair_conf(freq_items_dict, pairs_dict):
    confidence = {}
    for pair in pairs_dict.keys():
        confidence[pair] = (pairs_dict[pair] / freq_items_dict[pair[0]])
        confidence[pair[1], pair[0]] = (pairs_dict[pair] / freq_items_dict[pair[1]])
    sorted_pairs_conf = list(confidence.items())
    sorted_pairs_conf.sort(key = lambda x:(-x[1], x[0][0], x[0][1]))
    return(sorted_pairs_conf)

# pairs_dict: dictionary containing frequent pairs
# triples_dict: dictionary containing frequent triples
# return: list of sorted confidence levels for frequent triples
def triples_conf(pairs_dict, triples_dict):
    confidence = {}
    for triple in triples_dict.keys():
        confidence[triple] = (triples_dict[triple] / pairs_dict[(triple[0], triple[1])])
        confidence[(triple[0], triple[2], triple[1])] = (triples_dict[triple] / pairs_dict[(triple[0], triple[2])])
        confidence[(triple[1], triple[2], triple[0])] = (triples_dict[triple] / pairs_dict[(triple[1], triple[2])])
    sorted_triples_conf = list(confidence.items())
    sorted_triples_conf.sort(key = lambda x:(-x[1], x[0][0], x[0][1], x[0][2]))
    return(sorted_triples_conf)

# freq_items_dict: dictionary containing frequent items
# pairs_dict: dictionary containing frequent pairs
# triples_dict: dictionary containing frequent triples
# n: number of items to be listed in final output
# return: none
def write_output(pairs_conf_list, triples_conf_list, n):
    with open('output.txt', 'w') as output_file:
        output_file.write('OUTPUT A\n')
        for i in range(0, n):
                line = (str(pairs_conf_list[i][0][0]) + ' ' 
                + str(pairs_conf_list[i][0][1]) + ' ' 
                + str(pairs_conf_list[i][1]) + '\n')
                output_file.write(line)

        output_file.write('OUTPUT B\n')
        for i in range(0, n):
                line = (str(triples_conf_list[i][0][0]) + ' ' 
                + str(triples_conf_list[i][0][1]) + ' ' 
                + str(triples_conf_list[i][0][2]) + ' ' 
                + str(triples_conf_list[i][1]) + '\n')
                output_file.write(line)
    return

##############################################################
# read input from specified file
baskets = read_input('browsing-data.txt')

# find frequent items in set
items = find_frequent_items(baskets)

# find frequent pairs and confidence
pairs = freq_k_items(baskets, 2)
pairs_conf_list = pair_conf(items, pairs)

# find frequent triples and confidence
triples = freq_k_items(baskets, 3)
triples_conf_list = triples_conf(pairs, triples)

# write results to output file
write_output(pairs_conf_list, triples_conf_list, 5)
##############################################################