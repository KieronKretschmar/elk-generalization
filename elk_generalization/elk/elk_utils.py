import torch
from distutils.util import strtobool


key_to_column = {
    "pi": "persona_introduceds",
    "pr": "persona_respondss",
    "ql": "quirky_labels",
    "ol": "objective_labels"
}
column_to_key = {v:k for k,v in key_to_column.items()}

class SplitConfig():
    """Defines training configuration for a split of a dataset for which certain labels are aligned or constant.
    """

    @classmethod
    def from_descriptor(cls, descriptor):

        for outer_part in descriptor.split("-"):
            if outer_part.startswith("label="):
                label_key = outer_part.split("=")[-1]
            elif outer_part.startswith("pa"):
                positively_aligned_keys = outer_part.split("_")[1:]
            elif outer_part.startswith("na"):
                negatively_aligned_keys = outer_part.split("_")[1:]
            elif outer_part.startswith("filters"):
                # Split the string into 'key=value' strings
                key_value_pairs = outer_part.split('_')[1:]

                # Iterate through the key-value pairs
                filter_keys = []
                filter_values = []
                for pair in key_value_pairs:
                    key, value = pair.split('=')
                    filter_keys.append(key)
                    filter_values.append(bool(strtobool(value)))
            else:
                raise ValueError("Can't parse descriptor for SplitConfig: ", descriptor)
        

        return cls(
            label_col=key_to_column[label_key],
            positively_aligned_cols=[key_to_column[k] for k in positively_aligned_keys],
            negatively_aligned_cols=[key_to_column[k] for k in negatively_aligned_keys],
            filter_cols=[key_to_column[k] for k in filter_keys],
            filter_values=filter_values
        )
    
    def __init__(self, label_col, positively_aligned_cols, negatively_aligned_cols, filter_cols, filter_values):
        self.label_col = label_col
        self.positively_aligned_cols = positively_aligned_cols
        self.negatively_aligned_cols = negatively_aligned_cols
        self.filter_cols = filter_cols
        self.filter_values = filter_values
        assert len(self.filter_cols) == len(self.filter_values), "Mismatch in number of filter criteria"
        
        self.key_to_column = key_to_column
        self.column_to_key = column_to_key

        self.label_key = self.column_to_key[label_col]
        self.positively_aligned_keys = [self.column_to_key[c] for c in positively_aligned_cols]
        self.negatively_aligned_keys = [self.column_to_key[c] for c in negatively_aligned_cols]
        self.filter_keys = [self.column_to_key[c] for c in filter_cols]
    
        self.filter_dict = {self.filter_keys[i]:self.filter_values[i] for i in range(len(self.filter_keys))}

    def descriptor(self):
        """Returns string containing all args in a particular format"""

        elements = [f"label={self.label_key}"]
        
        positively_aligned_keys = [column_to_key[col] for col in self.positively_aligned_cols]
        elements.append('_'.join(["pa"] + sorted(positively_aligned_keys)))
        negatively_aligned_keys = [column_to_key[col] for col in self.negatively_aligned_cols]
        elements.append('_'.join(["na"] + sorted(negatively_aligned_keys)))
        filters = []
        for i in range(len(self.filter_cols)):
            filter_key = self.column_to_key[self.filter_cols[i]]
            filters.append(f"{filter_key}={self.filter_values[i]}")
        elements.append('_'.join(["filters"] + sorted(filters)))
        
        return '-'.join(elements)
    
    def contains_segment(self, segment_cfg):
        """
        Compares the segment's value for label_col with the value of other labels it was filtered by.
        If cfg has at least one key in positively_aligned_key with a different value, or one in negatively_aligned_key with the same value, it returns False.
        If cfg does not have the value specified in filter_values for every key in filter_key, it returns False.
        Otherwise, it returns True.
        """

        # print(f"Checking if Split config {self.descriptor()} allows {segment_cfg.descriptor()}...")
        # If the label_key is not in the cfg, none of the other parameters have been aligned to it in this segment
        # (this is a convention, when creating the dataset)
        if self.label_key not in segment_cfg.filter_keys:
            # print(f"{self.label_key=} not in cfg")
            return False
        
        label_val = segment_cfg.filter_dict[self.label_key]

        for key in self.positively_aligned_keys:
            if (key not in segment_cfg.filter_dict) or (segment_cfg.filter_dict[key] != label_val):
                # print(f"{key=} not in segment cfg or is not positively aligned")
                return False
            
        for key in self.negatively_aligned_keys:
            if key in segment_cfg.filter_dict:
                assert type(segment_cfg.filter_dict[key]) is bool, "Negative alignment only works for boolean values"
            if (key not in segment_cfg.filter_dict) or (segment_cfg.filter_dict[key] == label_val):
                # print(f"{key=} not in segment cfg or is not negatively aligned")
                return False

        for key, value in self.filter_dict.items():
            if (key not in segment_cfg.filter_dict) or (segment_cfg.filter_dict[key] != value):
                # print(f"{key=} not in segment cfg or does not correspond to filter expectation {value=}")
                return False

        # print("Yes it does.")
        return True
    
    
    def identify_valid_segments(self, segment_configs):
        valid_cfgs = [cfg for cfg in segment_configs if self.segment_is_valid(cfg)]
        return valid_cfgs
    
class SegmentConfig():
    """Defines a segment of a dataset for which certain labels are constant.
    """
    @classmethod
    def from_descriptor(cls, descriptor):
        # Split the string into key-value pairs
        key_value_pairs = descriptor.split('_')

        # Iterate through the key-value pairs
        filter_cols = []
        filter_values = []
        for pair in key_value_pairs:
            key, value = pair.split('=')
            filter_cols.append(key_to_column[key])
            filter_values.append(bool(strtobool(value)))

        return cls(filter_cols=filter_cols, filter_values=filter_values)
    
    def __init__(self, filter_cols, filter_values):
        self.filter_cols = filter_cols
        self.filter_values = filter_values

        self.key_to_column = key_to_column
        self.column_to_key = column_to_key

        self.filter_keys = [self.column_to_key[c] for c in filter_cols]

        self.filter_dict = {self.filter_keys[i]:self.filter_values[i] for i in range(len(self.filter_keys))}

    def descriptor(self):
        filter_items = []
        for i, filter_key in enumerate(self.filter_keys):
            filter_items.append(f"{filter_key}={self.filter_values[i]}")
        return '_'.join(filter_items)


def aggregate_segments(paths, label_cols, reporter, device, data_split, log_odds_split_descriptor=None):
    out = {}
    for i, path in enumerate(paths):
        path = path / data_split
        hiddens_file = (
            "ccs_hiddens.pt"
            if reporter in {"ccs", "crc", "lr-on-pair"}
            else "hiddens.pt"
        )
        train_hiddens = torch.load(path / hiddens_file)
        train_n = train_hiddens[0].shape[0]
        d = train_hiddens[0].shape[-1]
        assert all(
            h.shape[0] == train_n for h in train_hiddens
        ), "Mismatched number of samples"
        assert all(h.shape[-1] == d for h in train_hiddens), "Mismatched hidden size"

        if log_odds_split_descriptor:
            log_odds_path = path / log_odds_split_descriptor / f"{reporter}_log_odds.pt"
            log_odds = torch.load(log_odds_path, map_location="cpu")

        if i==0:
            out["hiddens"] = train_hiddens
            if log_odds_split_descriptor:
                out["reporter_log_odds"] = log_odds
            for label_col in label_cols:
                out[label_col] = torch.load(path / f"{label_col}.pt").to(device).int()
        else:
            out["hiddens"] = [torch.cat([out["hiddens"][i], train_hiddens[i]]) for i in range(len(train_hiddens))]
            if log_odds_split_descriptor:
                out["reporter_log_odds"] = torch.cat([out["reporter_log_odds"], log_odds], axis=1)
            for label_col in label_cols:
                labels = torch.load(path / f"{label_col}.pt").to(device).int()
                assert len(labels) == train_n, "Mismatched number of labels"
                out[label_col] = torch.cat([out[label_col], labels])
        
    return out