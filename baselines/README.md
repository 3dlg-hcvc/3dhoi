## 3DADN

For 3DADN, a fork of [original repository](https://github.com/JasonQSY/Articulation3D) is present in [3DADN/Articulations](3DADN/Articulation3D/). Refer to the [README](3DADN/Articulation3D/README.md) for setup instructions.

For our experiments, we used the pretrained model ([here](https://www.dropbox.com/s/50uderl5ynan2yt/model_0059999.pth?dl=0)) provided by 3DADN, which is trained on their Internet video dataset.

To get 3DADN results, use the following commands to get predicted mesh, origin and axis estimates for evaluation,
```
cd 3DADN/Articulations3D/articulations3d/
python tools/inference.py --config/config.yaml  --input /path/to/videos/list --out_path /path/to/output/folder --save-obj --webvis
```

Once the predictions are generated, refer back to the [evaluations](https://github.com/3dlg-hcvc/3dhoi#evaluation) section to generate results for these predictions. 

## D3DHOI

For D3DHOI optimization, follow the instructions on the [original repository](https://github.com/facebookresearch/d3d-hoi/tree/main/optimization).

Once you have the optimized results on the D3DHOI dataset, copy the contents of [d3dhoi](d3dhoi/) folder into the repo and use the following command to get the predicted mesh, origin and axis estimates for evaluation,

```
python get_preds.py --data_folder /path/to/data/ \
                    --pred_folder /path/to/results/ \
                    --cad_folder /path/to/processed_cads/ \
                    --out_folder /path/to/predictions
```

Once the predictions are generated, refer back to the [evaluations](https://github.com/3dlg-hcvc/3dhoi#evaluation) section to generate results for these predictions. 

## LASR and ViSER

For [LASR](https://github.com/google/lasr) and [ViSER](https://github.com/gengshan-y/viser-release), we use the open-source code to train both on D3DHOI dataset. Refer to the original repositories for setup and instructions to run on new data. 

Once you have the optimized results on the D3DHOI dataset, copy the contents of [lasr_viser](lasr_viser/) folder into the repo and use the following command to get the predicted mesh, origin and axis estimates for evaluation,

```
python get_preds.py --pred_folder /path/to/results/ \
                    --videos_file /path/to/list/of/videos/ \
                    --method [lasr or viser]
```

Once the predictions are generated, refer back to the [evaluations](https://github.com/3dlg-hcvc/3dhoi#evaluation) section to generate results for these predictions. 

## Ditto

For Ditto, refer to the [original repository](https://github.com/UT-Austin-RPL/Ditto) for setup instructions.

For our experiments, we use the pretrained model trained on Shape2Motion dataset which can be found [here](https://utexas.box.com/s/a4h001b3ciicrt3f71t4xd3wjsm04be7). 

Once you have the code and model setup, copy the contents of the folder [ditto](ditto/) into the original repository and run the following command to get predicted mesh, origin and axis estimates for evaluation,

```
python inference.py --data_path /path/to/data/ \
                    --videos_file /path/to/list/of/videos/ \
                    --out_path /path/to/output/estimates 
```

To get a consistent single estimate of the origin and axis run,
```
python get_consistent_motion.py --data_path /path/to/data/ \
                                --videos_file /path/to/list/of/videos/ \
                                --res_path /path/to/output/estimates 
```

Once the predictions are generated, refer back to the [evaluations](https://github.com/3dlg-hcvc/3dhoi#evaluation) section to generate results for these predictions. 