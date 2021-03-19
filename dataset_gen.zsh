# generate dataset configurations
echo "Generating dataset config files for IID"
python src/dataset/blicket.py --train_size 6000 --val_size 2000 --test_size 2000 --regime IID --output_dataset_dir ./ACRE_IID/config
echo "Generating dataset config files for Comp"
python src/dataset/blicket.py --train_size 6000 --val_size 2000 --test_size 2000 --regime Comp --output_dataset_dir ./ACRE_Comp/config
echo "Generating dataset config files for Sys"
python src/dataset/blicket.py --train_size 6000 --val_size 2000 --test_size 2000 --regime Sys --output_dataset_dir ./ACRE_Sys/config

# call blender to render
alias blender2.79="/home/chizhang/blender-2.79b/blender"
mkdir ./log
mkdir ./log/IID
mkdir ./log/Comp
mkdir ./log/Sys

echo "Start rendering for IID"
echo "Rendering training set"
(( round=0 ))
(( step=50 ))
for (( start_index=round*step; start_index<=6000-step; start_index+=step ))
do  
    echo "Start from $start_index"
    (( end_index=start_index+step ))
    blender2.79 --background -noaudio --python ./src/render/render_images.py -- --start_index $start_index --end_index $end_index --use_gpu 1 --dataset_json ./ACRE_IID/config/train.json --shape_color_material_combos_json ./src/render/data/all.json --split train --output_image_dir ./ACRE_IID/images/ --output_scene_dir ./ACRE_IID/scenes/ --output_scene_file ./ACRE_IID/ACRE_scenes.json --output_blend_dir ./ACRE_IID/blendfiles >> "./log/IID/train_round_$round.txt"
    (( round+=1 ))
done

echo "Rendering val set"
(( round=0 ))
(( step=50 ))
for (( start_index=round*step; start_index<=2000-step; start_index+=step ))
do  
    echo "Start from $start_index"
    (( end_index=start_index+step ))
    blender2.79 --background -noaudio --python ./src/render/render_images.py -- --start_index $start_index --end_index $end_index --use_gpu 1 --dataset_json ./ACRE_IID/config/val.json --shape_color_material_combos_json ./src/render/data/all.json --split val --output_image_dir ./ACRE_IID/images/ --output_scene_dir ./ACRE_IID/scenes/ --output_scene_file ./ACRE_IID/ACRE_scenes.json --output_blend_dir ./ACRE_IID/blendfiles >> "./log/IID/val_round_$round.txt"
    (( round+=1 ))
done

echo "Rendering test set"
(( round=0 ))
(( step=50 ))
for (( start_index=round*step; start_index<=2000-step; start_index+=step ))
do  
    echo "Start from $start_index"
    (( end_index=start_index+step ))
    blender2.79 --background -noaudio --python ./src/render/render_images.py -- --start_index $start_index --end_index $end_index --use_gpu 1 --dataset_json ./ACRE_IID/config/test.json --shape_color_material_combos_json ./src/render/data/all.json --split test --output_image_dir ./ACRE_IID/images/ --output_scene_dir ./ACRE_IID/scenes/ --output_scene_file ./ACRE_IID/ACRE_scenes.json --output_blend_dir ./ACRE_IID/blendfiles >> "./log/IID/test_round_$round.txt"
    (( round+=1 ))
done

echo "Start rendering for Comp"
echo "Rendering training set"
(( round=0 ))
(( step=50 ))
for (( start_index=round*step; start_index<=6000-step; start_index+=step ))
do  
    echo "Start from $start_index"
    (( end_index=start_index+step ))
    blender2.79 --background -noaudio --python ./src/render/render_images.py -- --start_index $start_index --end_index $end_index --use_gpu 1 --dataset_json ./ACRE_Comp/config/train.json --shape_color_material_combos_json ./src/render/data/attr_train.json --split train --output_image_dir ./ACRE_Comp/images/ --output_scene_dir ./ACRE_Comp/scenes/ --output_scene_file ./ACRE_Comp/ACRE_scenes.json --output_blend_dir ./ACRE_Comp/blendfiles >> "./log/Comp/train_round_$round.txt"
    (( round+=1 ))
done

echo "Rendering val set"
(( round=0 ))
(( step=50 ))
for (( start_index=round*step; start_index<=2000-step; start_index+=step ))
do  
    echo "Start from $start_index"
    (( end_index=start_index+step ))
    blender2.79 --background -noaudio --python ./src/render/render_images.py -- --start_index $start_index --end_index $end_index --use_gpu 1 --dataset_json ./ACRE_Comp/config/val.json --shape_color_material_combos_json ./src/render/data/attr_test.json --split val --output_image_dir ./ACRE_Comp/images/ --output_scene_dir ./ACRE_Comp/scenes/ --output_scene_file ./ACRE_Comp/ACRE_scenes.json --output_blend_dir ./ACRE_Comp/blendfiles >> "./log/Comp/val_round_$round.txt"
    (( round+=1 ))
done

echo "Rendering test set"
(( round=0 ))
(( step=50 ))
for (( start_index=round*step; start_index<=2000-step; start_index+=step ))
do  
    echo "Start from $start_index"
    (( end_index=start_index+step ))
    blender2.79 --background -noaudio --python ./src/render/render_images.py -- --start_index $start_index --end_index $end_index --use_gpu 1 --dataset_json ./ACRE_Comp/config/test.json --shape_color_material_combos_json ./src/render/data/attr_test.json --split test --output_image_dir ./ACRE_Comp/images/ --output_scene_dir ./ACRE_Comp/scenes/ --output_scene_file ./ACRE_Comp/ACRE_scenes.json --output_blend_dir ./ACRE_Comp/blendfiles >> "./log/Comp/test_round_$round.txt"
    (( round+=1 ))
done

echo "Start rendering for Sys"
echo "Rendering training set"
(( round=0 ))
(( step=50 ))
for (( start_index=round*step; start_index<=6000-step; start_index+=step ))
do  
    echo "Start from $start_index"
    (( end_index=start_index+step ))
    blender2.79 --background -noaudio --python ./src/render/render_images.py -- --start_index $start_index --end_index $end_index --use_gpu 1 --dataset_json ./ACRE_Sys/config/train.json --shape_color_material_combos_json ./src/render/data/all.json --split train --output_image_dir ./ACRE_Sys/images/ --output_scene_dir ./ACRE_Sys/scenes/ --output_scene_file ./ACRE_Sys/ACRE_scenes.json --output_blend_dir ./ACRE_Sys/blendfiles >> "./log/Sys/train_round_$round.txt"
    (( round+=1 ))
done

echo "Rendering val set"
(( round=0 ))
(( step=50 ))
for (( start_index=round*step; start_index<=2000-step; start_index+=step ))
do  
    echo "Start from $start_index"
    (( end_index=start_index+step ))
    blender2.79 --background -noaudio --python ./src/render/render_images.py -- --start_index $start_index --end_index $end_index --use_gpu 1 --dataset_json ./ACRE_Sys/config/val.json --shape_color_material_combos_json ./src/render/data/all.json --split val --output_image_dir ./ACRE_Sys/images/ --output_scene_dir ./ACRE_Sys/scenes/ --output_scene_file ./ACRE_Sys/ACRE_scenes.json --output_blend_dir ./ACRE_Sys/blendfiles >> "./log/Sys/val_round_$round.txt"
    (( round+=1 ))
done

echo "Rendering test set"
(( round=0 ))
(( step=50 ))
for (( start_index=round*step; start_index<=2000-step; start_index+=step ))
do  
    echo "Start from $start_index"
    (( end_index=start_index+step ))
    blender2.79 --background -noaudio --python ./src/render/render_images.py -- --start_index $start_index --end_index $end_index --use_gpu 1 --dataset_json ./ACRE_Sys/config/test.json --shape_color_material_combos_json ./src/render/data/all.json --split test --output_image_dir ./ACRE_Sys/images/ --output_scene_dir ./ACRE_Sys/scenes/ --output_scene_file ./ACRE_Sys/ACRE_scenes.json --output_blend_dir ./ACRE_Sys/blendfiles >> "./log/Sys/test_round_$round.txt"
    (( round+=1 ))
done