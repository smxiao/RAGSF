import argparse

def get_params():
    parser = argparse.ArgumentParser(description="RAGSF")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--plm_eval_mode", action="store_true")
    parser.add_argument("--model", type=str, default='t5')
    parser.add_argument("--model_name_or_path", default='t5-base')
    parser.add_argument("--model_saved_path", default='model_save/models')
    parser.add_argument("--model_name", default='PlayMusic_0')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--target_domain", type=str, default="PlayMusic")
    parser.add_argument("--dataset_dir",type=str,default="./retrieve_data/")
    parser.add_argument("--test_only",default=False,action="store_true")
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument("--use_cuda",default=True,action="store_false")
    parser.add_argument('--max_epochs', type=int, default=15)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument("--prefix_num_tokens", type=int, default=10)
    parser.add_argument("--prefix_dp", type=float, default=0.1)
    parser.add_argument("--n_samples", type=int, default=0)
    
    args = parser.parse_args()
    
    return args