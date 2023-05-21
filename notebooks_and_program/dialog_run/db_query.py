from collections import defaultdict
from dialogue_config import no_query_keys, usersim_default_key
import copy


class DBQuery:

    def __init__(self, database):

        self.database = database
        self.cached_db_slot = defaultdict(dict)
        self.cached_db = defaultdict(dict)
        self.no_query = no_query_keys
        self.match_key = usersim_default_key

    def fill_inform_slot(self, inform_slot_to_fill, current_inform_slots):

        assert len(inform_slot_to_fill) == 1

        key = list(inform_slot_to_fill.keys())[0]

        current_informs = copy.deepcopy(current_inform_slots)
        current_informs.pop(key, None)

        db_results = self.get_db_results(current_informs)

        filled_inform = {}
        values_dict = self._count_slot_values(key, db_results)
        if values_dict:
            filled_inform[key] = max(values_dict, key=values_dict.get)
        else:
            filled_inform[key] = 'no match available'

        return filled_inform

    def _count_slot_values(self, key, db_subdict):

        slot_values = defaultdict(int)
        for id in db_subdict.keys():
            current_option_dict = db_subdict[id]
            if key in current_option_dict.keys():
                slot_value = current_option_dict[key]
                slot_values[slot_value] += 1
        return slot_values

    def get_db_results(self, constraints):

        new_constraints = {k: v for k, v in constraints.items() if k not in self.no_query and v is not 'anything'}

        inform_items = frozenset(new_constraints.items())
        cache_return = self.cached_db[inform_items]

        if cache_return == None:
            return {}
        if cache_return:
            return cache_return

        available_options = {}
        for id in self.database.keys():
            current_option_dict = self.database[id]
            if len(set(new_constraints.keys()) - set(self.database[id].keys())) == 0:
                match = True
                for k, v in new_constraints.items():
                    if str(v).lower() != str(current_option_dict[k]).lower():
                        match = False
                if match:
                    self.cached_db[inform_items].update({id: current_option_dict})
                    available_options.update({id: current_option_dict})

        if not available_options:
            self.cached_db[inform_items] = None

        return available_options

    def get_db_results_for_slots(self, current_informs):

        inform_items = frozenset(current_informs.items())
        cache_return = self.cached_db_slot[inform_items]

        if cache_return:
            return cache_return

        db_results = {key: 0 for key in current_informs.keys()}
        db_results['matching_all_constraints'] = 0

        for id in self.database.keys():
            all_slots_match = True
            for CI_key, CI_value in current_informs.items():
                if CI_key in self.no_query:
                    continue
                if CI_value == 'anything':
                    db_results[CI_key] += 1
                    continue
                if CI_key in self.database[id].keys():
                    if CI_value.lower() == self.database[id][CI_key].lower():
                        db_results[CI_key] += 1
                    else:
                        all_slots_match = False
                else:
                    all_slots_match = False
            if all_slots_match: db_results['matching_all_constraints'] += 1

        self.cached_db_slot[inform_items].update(db_results)
        assert self.cached_db_slot[inform_items] == db_results
        return db_results
