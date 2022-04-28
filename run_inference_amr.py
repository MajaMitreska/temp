import os

def main(model_name, tokenizer='facebook/bart-large', gpu_id=0,
         datacate='examples', data_dir=None, train_data_file=None, eval_data_file=None, test_data_file=None):
  args = {}
  root_dir = os.getcwd()
  output_dir_name = 'outputs/Eval'+datacate+'-AMRBART'
  output_dir = os.path.join(root_dir, output_dir_name)
  cache_dir = os.path.join(root_dir, '..', '.cache')

  if not os.isdir(output_dir):
    os.mkdir(output_dir)
  else:
    print('The path already exists')
    os.rmdir(output_dir)
  
  args['CUDA_VISIBLE_DEVICES']=gpu_id
  args['data_dir']=data_dir
  args['train_data_file']=data_dir/'train.jsonl' 
  args['eval_data_file']=data_dir/'val.jsonl'
  args['test_data_file']=data_dir/'data4parsing.jsonl'
  args['model_type']=model_name
  args['model_name_or_path']=model_name
  args['tokenizer_name_or_path']=tokenizer
  args['val_metric']="smatch"
  args['learning_rate']=8e-6
  args['max_epochs']=1
  args['max_steps']=-1
  args['per_gpu_eval_batch_size']=48
  args['unified_input']="store_true"
  args['gpus']=1 
  args['output_dir']=output_dir
  args['cache_dir']=cache_dir
  args['num_sanity_val_steps']=0
  args['src_block_size']=256
  args['tgt_block_size']=512
  args['eval_max_length']=512
  args['eval_num_workers']=2
  args['process_num_workers']=8
  args['OMP_NUM_THREADS']=10
  args['do_eval']="store_true"
  args['seed']=42
  args['fp16']="store_true"
  args['eval_beam']=5
  args['src_block_size']=512,
  args['tgt_block_size']=512,
  args['eval_lenpen']=1.0,
  args['eval_max_length']=512,
  args['label_smoothing']=0.0
  args['dropout']=0

  args['smart_init']="store_true"
  args['do_train']="store_true"
  args['do_predict']="store_true"
  
  args['evaluate_during_training']="store_true"
    
  args['per_gpu_train_batch_size']=4,
  args['per_gpu_eval_batch_size']=4,
  args['train_num_workers']=4
  args['eval_num_workers']=4
  args['process_num_workers']=1
  args['early_stopping_patience']=0
  args['lr_scheduler']="linear"
  args['learning_rate']=5e-5
  args['weight_decay']=0.0
  args['adam_epsilon']=1e-8
  args['max_grad_norm']=1.0
  args['warmup_steps']=0
  args['fp16_opt_level']=="O2"
  args['save_total_limit']=1
  args['save_interval']=-1
  args['resume']="store_true"


  inference_amr(args)

