# ASGEA
## 🔬 Dependencies
```
pytorch 1.12.0
torch_geometric 2.2.0
torch_scatter 2.0.9
transformers 4.26.1
```

## 🚀 Train

- **Quick start**: Using script file for ASGEA-MM.

```bash
# FBDB15K & FBYG15K
>> bash run.sh
# DBP15K
>> bash run_dbp.sh
# Multi OpenEA
>> bash run_oea.sh
```

- **❗tips**: If you are using slurm, you can change the `.sh` file from

  ```bash
  datas="FBDB15K FBYG15K"
  rates="0.2 0.5 0.8"
  expn=$1
  if [ ! -d "results/${expn}" ]; then
    mkdir results/${expn}
  fi
  if [ ! -d "results/${expn}/backup" ]; then
    mkdir results/${expn}/backup
  fi
  cp *.py results/${expn}/backup/
  for data in $datas ; do
    for rate in $rates ; do
      python train.py --data_split norm --n_batch 4 --n_layer 5 --lr 0.001 --data_choice ${data} --data_rate ${rate} --exp_name ${expn} --mm 1 --img_dim 4096
      # echo "sbatch -o ${data}_${rate}.log run.slurm 4 5 0.001 ${data} ${rate}"
      # sbatch -o ${expn}_${data}_${rate}.log run.slurm 4 5 0.001 ${data} ${rate} ${expn}
    done
  done
  ```

  to

  ```bash
  datas="FBDB15K FBYG15K"
  rates="0.2 0.5 0.8"
  expn=$1
  if [ ! -d "results/${expn}" ]; then
    mkdir results/${expn}
  fi
  if [ ! -d "results/${expn}/backup" ]; then
    mkdir results/${expn}/backup
  fi
  cp *.py results/${expn}/backup/
  for data in $datas ; do
    for rate in $rates ; do
      echo "sbatch -o ${data}_${rate}.log run.slurm 4 5 0.001 ${data} ${rate}"
      sbatch -o ${expn}_${data}_${rate}.log run.slurm 4 5 0.001 ${data} ${rate} ${expn}
    done
  done
  ```

- **for ASGEA-Stru**:  Just set `mm=0`.


## 📚 Dataset

```
ROOT
├── data
│   └── mmkg
└── ASGEA
```

#### Code Path


```
ASGEA
├── base_model.py
├── data.py
├── load_data.py
├── models.py
├── opt.py
├── README.md
├── run.sh
├── run.slurm
├── run_dbp.sh
├── run_dbp.slurm
├── run_oea.sh
├── run_oea.slurm
├── train.py
├── utils.py
└── vis.py
```

</details>

#### Data Path

```
mmkg                                                         
├─ DBP15K                                                    
│  ├─ fr_en                                                  
│  │  ├─ att_features100.npy                                 
│  │  ├─ att_features500.npy                                 
│  │  ├─ att_rel_features100.npy                             
│  │  ├─ att_rel_features500.npy                             
│  │  ├─ att_val_features100.npy                             
│  │  ├─ att_val_features500.npy                             
│  │  ├─ en_att_triples                                      
│  │  ├─ ent_ids_1                                           
│  │  ├─ ent_ids_2                                           
│  │  ├─ fr_att_triples                                      
│  │  ├─ ill_ent_ids                                         
│  │  ├─ training_attrs_1                                    
│  │  ├─ training_attrs_2                                    
│  │  ├─ triples_1                                           
│  │  └─ triples_2                                           
│  ├─ ja_en                                                  
│  │  ├─ att_features100.npy                                 
│  │  ├─ att_features500.npy                                 
│  │  ├─ att_rel_features100.npy                             
│  │  ├─ att_rel_features500.npy                             
│  │  ├─ att_val_features100.npy                             
│  │  ├─ att_val_features500.npy                             
│  │  ├─ en_att_triples                                      
│  │  ├─ ent_ids_1                                           
│  │  ├─ ent_ids_2                                           
│  │  ├─ ill_ent_ids                                         
│  │  ├─ ja_att_triples                                      
│  │  ├─ training_attrs_1                                    
│  │  ├─ training_attrs_2                                    
│  │  ├─ triples_1                                           
│  │  └─ triples_2                                           
│  ├─ translated_ent_name                                    
│  │  ├─ dbp_fr_en.json                                      
│  │  ├─ dbp_ja_en.json                                      
│  │  └─ dbp_zh_en.json                                      
│  └─ zh_en                                                  
│     ├─ att_features100.npy                                 
│     ├─ att_features500.npy                                 
│     ├─ att_rel_features100.npy                             
│     ├─ att_rel_features500.npy                             
│     ├─ att_val_features100.npy                             
│     ├─ att_val_features500.npy                             
│     ├─ en_att_triples                                      
│     ├─ ent_ids_1                                           
│     ├─ ent_ids_2                                           
│     ├─ ill_ent_ids                                         
│     ├─ rule_test.txt                                       
│     ├─ rule_train.txt                                      
│     ├─ training_attrs_1                                    
│     ├─ training_attrs_2                                    
│     ├─ triples_1                                           
│     ├─ triples_2                                           
│     └─ zh_att_triples                                      
├─ FBDB15K                                                   
│  └─ norm                                                   
│     ├─ DB15K_NumericalTriples.txt                          
│     ├─ FB15K_NumericalTriples.txt                          
│     ├─ att_features.npy                                    
│     ├─ att_rel_features.npy                                
│     ├─ att_val_features.npy                                
│     ├─ ent_ids_1                                           
│     ├─ ent_ids_2                                           
│     ├─ fbid2name.txt                                       
│     ├─ id2relation.txt                                     
│     ├─ ill_ent_ids                                         
│     ├─ training_attrs_1                                    
│     ├─ training_attrs_2                                    
│     ├─ triples_1                                           
│     └─ triples_2                                           
├─ FBYG15K                                                   
│  └─ norm                                                   
│     ├─ FB15K_NumericalTriples.txt                          
│     ├─ YAGO15K_NumericalTriples.txt                        
│     ├─ att_features.npy                                    
│     ├─ att_rel_features.npy                                
│     ├─ att_val_features.npy                                
│     ├─ ent_ids_1                                           
│     ├─ ent_ids_2                                           
│     ├─ fbid2name.txt                                       
│     ├─ id2relation.txt                                     
│     ├─ ill_ent_ids                                         
│     ├─ training_attrs_1                                    
│     ├─ training_attrs_2                                    
│     ├─ triples_1                                           
│     └─ triples_2                                           
├─ MEAformer                                                 
├─ OpenEA                                                    
│  ├─ OEA_D_W_15K_V1                                         
│  │  ├─ att_features.npy                                    
│  │  ├─ att_features500.npy                                 
│  │  ├─ att_rel_features.npy                                
│  │  ├─ att_rel_features500.npy                             
│  │  ├─ att_val_features.npy                                
│  │  ├─ att_val_features500.npy                             
│  │  ├─ attr_triples_1                                      
│  │  ├─ attr_triples_2                                      
│  │  ├─ ent_ids_1                                           
│  │  ├─ ent_ids_2                                           
│  │  ├─ ill_ent_ids                                         
│  │  ├─ rel_ids                                             
│  │  ├─ training_attrs_1                                    
│  │  ├─ training_attrs_2                                    
│  │  ├─ triples_1                                           
│  │  └─ triples_2                                           
│  ├─ OEA_D_W_15K_V2                                         
│  │  ├─ att_features.npy                                    
│  │  ├─ att_features500.npy                                 
│  │  ├─ att_rel_features.npy                                
│  │  ├─ att_rel_features500.npy                             
│  │  ├─ att_val_features.npy                                
│  │  ├─ att_val_features500.npy                             
│  │  ├─ attr_triples_1                                      
│  │  ├─ attr_triples_2                                      
│  │  ├─ ent_ids_1                                           
│  │  ├─ ent_ids_2                                           
│  │  ├─ ill_ent_ids                                         
│  │  ├─ rel_ids                                             
│  │  ├─ training_attrs_1                                    
│  │  ├─ training_attrs_2                                    
│  │  ├─ triples_1                                           
│  │  └─ triples_2                                           
│  ├─ OEA_D_Y_15K_V1                                         
│  │  ├─ 721_5fold                                           
│  │  │  ├─ 1                                                
│  │  │  │  ├─ test_links                                    
│  │  │  │  ├─ train_links                                   
│  │  │  │  └─ valid_links                                   
│  │  │  ├─ 2                                                
│  │  │  │  ├─ test_links                                    
│  │  │  │  ├─ train_links                                   
│  │  │  │  └─ valid_links                                   
│  │  │  ├─ 3                                                
│  │  │  │  ├─ test_links                                    
│  │  │  │  ├─ train_links                                   
│  │  │  │  └─ valid_links                                   
│  │  │  ├─ 4                                                
│  │  │  │  ├─ test_links                                    
│  │  │  │  ├─ train_links                                   
│  │  │  │  └─ valid_links                                   
│  │  │  └─ 5                                                
│  │  │     ├─ test_links                                    
│  │  │     ├─ train_links                                   
│  │  │     └─ valid_links                                   
│  │  ├─ attr_triples_1                                      
│  │  ├─ attr_triples_2                                      
│  │  ├─ ent_ids_1                                           
│  │  ├─ ent_ids_2                                           
│  │  ├─ ent_links                                           
│  │  ├─ ill_ent_ids                                         
│  │  ├─ rel_ids                                             
│  │  ├─ rel_triples_1                                       
│  │  ├─ rel_triples_2                                       
│  │  ├─ triples_1                                           
│  │  └─ triples_2                                           
│  ├─ OEA_D_Y_15K_V2                                         
│  │  ├─ 721_5fold                                           
│  │  │  ├─ 1                                                
│  │  │  │  ├─ test_links                                    
│  │  │  │  ├─ train_links                                   
│  │  │  │  └─ valid_links                                   
│  │  │  ├─ 2                                                
│  │  │  │  ├─ test_links                                    
│  │  │  │  ├─ train_links                                   
│  │  │  │  └─ valid_links                                   
│  │  │  ├─ 3                                                
│  │  │  │  ├─ test_links                                    
│  │  │  │  ├─ train_links                                   
│  │  │  │  └─ valid_links                                   
│  │  │  ├─ 4                                                
│  │  │  │  ├─ test_links                                    
│  │  │  │  ├─ train_links                                   
│  │  │  │  └─ valid_links                                   
│  │  │  └─ 5                                                
│  │  │     ├─ test_links                                    
│  │  │     ├─ train_links                                   
│  │  │     └─ valid_links                                   
│  │  ├─ attr_triples_1                                      
│  │  ├─ attr_triples_2                                      
│  │  ├─ ent_ids_1                                           
│  │  ├─ ent_ids_2                                           
│  │  ├─ ent_links                                           
│  │  ├─ ill_ent_ids                                         
│  │  ├─ rel_ids                                             
│  │  ├─ rel_triples_1                                       
│  │  ├─ rel_triples_2                                       
│  │  ├─ triples_1                                           
│  │  └─ triples_2                                           
│  ├─ OEA_EN_DE_15K_V1                                       
│  │  ├─ att_features.npy                                    
│  │  ├─ att_features500.npy                                 
│  │  ├─ att_rel_features.npy                                
│  │  ├─ att_rel_features500.npy                             
│  │  ├─ att_val_features.npy                                
│  │  ├─ att_val_features500.npy                             
│  │  ├─ attr_triples_1                                      
│  │  ├─ attr_triples_2                                      
│  │  ├─ ent_ids_1                                           
│  │  ├─ ent_ids_2                                           
│  │  ├─ ill_ent_ids                                         
│  │  ├─ rel_ids                                             
│  │  ├─ training_attrs_1                                    
│  │  ├─ training_attrs_2                                    
│  │  ├─ triples_1                                           
│  │  └─ triples_2                                           
│  ├─ OEA_EN_DE_15K_V2                                       
│  │  ├─ 721_5fold                                           
│  │  │  ├─ 1                                                
│  │  │  │  ├─ test_links                                    
│  │  │  │  ├─ train_links                                   
│  │  │  │  └─ valid_links                                   
│  │  │  ├─ 2                                                
│  │  │  │  ├─ test_links                                    
│  │  │  │  ├─ train_links                                   
│  │  │  │  └─ valid_links                                   
│  │  │  ├─ 3                                                
│  │  │  │  ├─ test_links                                    
│  │  │  │  ├─ train_links                                   
│  │  │  │  └─ valid_links                                   
│  │  │  ├─ 4                                                
│  │  │  │  ├─ test_links                                    
│  │  │  │  ├─ train_links                                   
│  │  │  │  └─ valid_links                                   
│  │  │  └─ 5                                                
│  │  │     ├─ test_links                                    
│  │  │     ├─ train_links                                   
│  │  │     └─ valid_links                                   
│  │  ├─ attr_triples_1                                      
│  │  ├─ attr_triples_2                                      
│  │  ├─ ent_ids_1                                           
│  │  ├─ ent_ids_2                                           
│  │  ├─ ent_links                                           
│  │  ├─ ill_ent_ids                                         
│  │  ├─ rel_ids                                             
│  │  ├─ rel_triples_1                                       
│  │  ├─ rel_triples_2                                       
│  │  ├─ triples_1                                           
│  │  └─ triples_2                                           
│  ├─ OEA_EN_FR_15K_V1                                       
│  │  ├─ att_features.npy                                    
│  │  ├─ att_rel_features.npy                                
│  │  ├─ att_val_features.npy                                
│  │  ├─ attr_triples_1                                      
│  │  ├─ attr_triples_2                                      
│  │  ├─ ent_ids_1                                           
│  │  ├─ ent_ids_2                                           
│  │  ├─ ill_ent_ids                                         
│  │  ├─ rel_ids                                             
│  │  ├─ training_attrs_1                                    
│  │  ├─ training_attrs_2                                    
│  │  ├─ triples_1                                           
│  │  └─ triples_2                                           
│  ├─ OEA_EN_FR_15K_V2                                       
│  │  ├─ 721_5fold                                           
│  │  │  ├─ 1                                                
│  │  │  │  ├─ test_links                                    
│  │  │  │  ├─ train_links                                   
│  │  │  │  └─ valid_links                                   
│  │  │  ├─ 2                                                
│  │  │  │  ├─ test_links                                    
│  │  │  │  ├─ train_links                                   
│  │  │  │  └─ valid_links                                   
│  │  │  ├─ 3                                                
│  │  │  │  ├─ test_links                                    
│  │  │  │  ├─ train_links                                   
│  │  │  │  └─ valid_links                                   
│  │  │  ├─ 4                                                
│  │  │  │  ├─ test_links                                    
│  │  │  │  ├─ train_links                                   
│  │  │  │  └─ valid_links                                   
│  │  │  └─ 5                                                
│  │  │     ├─ test_links                                    
│  │  │     ├─ train_links                                   
│  │  │     └─ valid_links                                   
│  │  ├─ attr_triples_1                                      
│  │  ├─ attr_triples_2                                      
│  │  ├─ ent_ids_1                                           
│  │  ├─ ent_ids_2                                           
│  │  ├─ ent_links                                           
│  │  ├─ ill_ent_ids                                         
│  │  ├─ rel_ids                                             
│  │  ├─ rel_triples_1                                       
│  │  ├─ rel_triples_2                                       
│  │  ├─ triples_1                                           
│  │  └─ triples_2                                           
│  ├─ pkl                                                    
│  │  ├─ OEA_D_W_15K_V1_id_img_feature_dict.pkl              
│  │  ├─ OEA_D_W_15K_V2_id_img_feature_dict.pkl              
│  │  ├─ OEA_EN_DE_15K_V1_id_img_feature_dict.pkl            
│  │  └─ OEA_EN_FR_15K_V1_id_img_feature_dict.pkl       
│  └─ data.py                                                
├─ dump                                                      
├─ embedding                                                 
│  ├─ dbp_fr_en_char.pkl                                     
│  ├─ dbp_fr_en_name.pkl                                     
│  ├─ dbp_ja_en_char.pkl                                     
│  ├─ dbp_ja_en_name.pkl                                     
│  ├─ dbp_zh_en_char.pkl                                     
│  ├─ dbp_zh_en_name.pkl                                     
│  └─ glove.6B.300d.txt                                      
└─ pkls                                                      
   ├─ FBDB15K_id_img_feature_dict.pkl                        
   ├─ FBYG15K_id_img_feature_dict.pkl                        
   ├─ dbpedia_wikidata_15k_dense_GA_id_img_feature_dict.pkl  
   ├─ dbpedia_wikidata_15k_norm_GA_id_img_feature_dict.pkl   
   ├─ fr_en_GA_id_img_feature_dict.pkl                       
   ├─ ja_en_GA_id_img_feature_dict.pkl                       
   └─ zh_en_GA_id_img_feature_dict.pkl                       
```

</details>