import re
import torch


def split_all_chains(chains):
        return [split_chain(chain) for chain in chains]


def split_chain(chain):

    # Add number to split reasoning chains unsing listings
    numbers = [str(n)+". " for n in range(20)]
    # Use regular expression to split the string
    delimiters = ['.\n', '. ', '? ', '! ', '\n'] + numbers

    regex_pattern = '|'.join(map(re.escape, delimiters))
    parts = re.split(regex_pattern, chain)

    # Remove empty strings from the result
    parts = [part.strip() for part in parts if part.strip()]

    return parts


def calculate_means(dict_list, keys):
        means_dict = dict()
        # Calculate the mean for each specified key
        for key in keys:
            values = torch.tensor([d[key] for d in dict_list])
            mean = torch.mean(values)
            means_dict[key] = mean.item()

        return means_dict