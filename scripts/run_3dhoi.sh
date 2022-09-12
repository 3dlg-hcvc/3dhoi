for x in `cat $1`; do
    python main.py --data_path data/d3dhoi_video_data/$x --seed 101 --iterations 500 --out_path results/cubeopt/$x/ --step 2 --desc ""
done;
