class Tournament:
    def __init__(self, num_entries=1024, ghr_bits=10):
        self.num_entries = num_entries
        self.ghr_bits = ghr_bits

        # Global predictor
        self.global_history = 0  #(1 for T, 0 for NT)
        self.global_pht = [1] * num_entries  # 2-bit saturating counters, init to weakly not-taken

        # Local predictor
        self.local_counters = [1] * num_entries  # One 2-bit counter per PC index
        self.local_btb = [None] * num_entries    # BTB

        # Selector: 2-bit counters
        # 0 or 1 => favor local; 2 or 3 => favor global
        self.choice_table = [0] * num_entries  #strongly local

    def saturating_inc(self, val):
        return min(val + 1, 3)

    def saturating_dec(self, val):
        return max(val - 1, 0)

    def index_global(self, pc):
        #(PC >> 3) XOR global_history, masked to the table size.
        pc_shifted = pc >> 3 
        return (pc_shifted ^ self.global_history) & (self.num_entries - 1) 

    def index_local(self, pc):
        return pc % self.num_entries

    def predict(self, pc):
        g_index = self.index_global(pc)
        l_index = self.index_local(pc)

        global_counter = self.global_pht[g_index]
        local_counter  = self.local_counters[l_index]
        choice_counter = self.choice_table[g_index]

        # 2-bit counters: 2 or 3 means predict taken.
        global_prediction = (global_counter >= 2)
        local_prediction  = (local_counter >= 2)

        if choice_counter < 2:
            prediction = local_prediction
            used = "local"
        else:
            prediction = global_prediction
            used = "global"

        predicted_target = None
        if used == "local" and local_prediction:
            predicted_target = self.local_btb[l_index]

        return prediction, used, predicted_target

    def update(self, pc, actual_taken, actual_target=None):
        g_index = self.index_global(pc)
        l_index = self.index_local(pc)

        pred_taken, used, _ = self.predict(pc)

        # Update global predictor
        if actual_taken:
            self.global_pht[g_index] = self.saturating_inc(self.global_pht[g_index])
        else:
            self.global_pht[g_index] = self.saturating_dec(self.global_pht[g_index])

        # Update local
        if actual_taken:
            self.local_counters[l_index] = self.saturating_inc(self.local_counters[l_index])
        else:
            self.local_counters[l_index] = self.saturating_dec(self.local_counters[l_index])

        # Update local BTB if branch is taken
        if actual_taken:
            self.local_btb[l_index] = actual_target

        # Update selector (choice table)
        global_correct = ((self.global_pht[g_index] >= 2) == actual_taken)
        local_correct  = ((self.local_counters[l_index] >= 2) == actual_taken)
        if global_correct and not local_correct:
            self.choice_table[g_index] = self.saturating_inc(self.choice_table[g_index])
        elif local_correct and not global_correct:
            self.choice_table[g_index] = self.saturating_dec(self.choice_table[g_index])

        # update the global history register
        self.global_history = ((self.global_history << 1) & ((1 << self.ghr_bits) - 1)) | (1 if actual_taken else 0)

    def print_debug_info(self, pc):
        """Optional: Print the internal state for a given PC for debugging."""
        g_index = self.index_global(pc)
        l_index = self.index_local(pc)
        print(f"PC=0x{pc:X}")
        print(f"  Global History: {bin(self.global_history)}")
        print(f"  GShare Index: {g_index}")
        print(f"  Local Index: {l_index}")
        print(f"  Global PHT Counter: {self.global_pht[g_index]}")
        print(f"  Local Counter: {self.local_counters[l_index]}")
        print(f"  Selector Counter: {self.choice_table[g_index]}")
        print(f"  Local BTB Target: {self.local_btb[l_index]}")
        print()

# Example usage:
if __name__ == "__main__":
    predictor = Tournament(num_entries=16, ghr_bits=4)
    test_pc = 0x1000

    # Predict before any update
    pred, used, target = predictor.predict(test_pc)
    print(f"Initial prediction: taken={pred}, predictor used: {used}, target={target}")

    # Update the predictor assuming the branch was taken with target 0x2000
    predictor.update(test_pc, True, actual_target=0x2000)
    predictor.print_debug_info(test_pc)