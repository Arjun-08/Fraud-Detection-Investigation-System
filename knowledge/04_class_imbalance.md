# Class Imbalance
When legitimate transactions greatly outnumber fraud, high accuracy can coexist with poor fraud recall. The project uses scale_pos_weight = N_negative / N_positive to increase the training importance of the minority class without creating synthetic transactions.
