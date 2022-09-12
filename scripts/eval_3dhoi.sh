python evals/eval_reconstruction_avg.py --videos_file data/3dhoi_videos.txt
                                        --gt_dir data/d3dhoi_video_data/
                                        --pred_dir results/cubeopt/ 
                                        --method cubeopt 
                                        --out_dir results/cubeopt/errors


python evals/eval_motion.py --videos_file data/3dhoi_videos.txt
                            --gt_dir data/d3dhoi_video_data/
                            --pred_dir results/cubeopt/ 
                            --method cubeopt 
                            --out_dir results/cubeopt/errors

python evals/compile_inter.py --result_dir results/cubeopt/errors

python evals/eval_accuracy.py --result_dir results/cubeopt/errors