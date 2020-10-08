python3 main.py --pc 8 --nets 1 --primary-unit-size 800 > f8_1.txt
mv results/trained_model/model_epoch_15.pth results/trained_model/model_epoch_15_8_1.pth
python3 main.py --pc 16 --nets 1 --primary-unit-size 1600 > f16_1.txt
mv results/trained_model/model_epoch_15.pth results/trained_model/model_epoch_15_16_1.pth
python3 main.py --pc 32 --nets 1 --primary-unit-size 3200 > f32_1.txt
mv results/trained_model/model_epoch_15.pth results/trained_model/model_epoch_15_32_1.pth


python3 main.py --pc 8 --nets 2 --primary-unit-size 968 > f8_2.txt
mv results/trained_model/model_epoch_15.pth  results/trained_model/model_epoch_15_8_2.pth
python3 main.py --pc 16 --nets 2 --primary-unit-size 1936 > f16_2.txt
mv results/trained_model/model_epoch_15.pth  results/trained_model/model_epoch_15_16_2.pth
python3 main.py --pc 32 --nets 2 --primary-unit-size 3872 > f32_2.txt
mv     results/trained_model/model_epoch_15.pth  results/trained_model/model_epoch_15_32_2.pth

python3 main.py --pc 8 --nets 3 --primary-unit-size 968 > f8_3.txt
mv     results/trained_model/model_epoch_15.pth  results/trained_model/model_epoch_15_8_3.pth
python3 main.py --pc 16 --nets 3 --primary-unit-size 1936 > f16_3.txt
mv     results/trained_model/model_epoch_15.pth  results/trained_model/model_epoch_15_16_3.pth
python3 main.py --pc 32 --nets 3 --primary-unit-size 3872 > f32_3.txt
mv     results/trained_model/model_epoch_15.pth  results/trained_model/model_epoch_15_32_3.pth

python3 main.py --pc 8 --nets 4 --primary-unit-size 2048 > f8_4.txt
mv     results/trained_model/model_epoch_15.pth  results/trained_model/model_epoch_15_8_4.pth
python3 main.py --pc 16 --nets 4 --primary-unit-size 4096 > f16_4.txt
mv     results/trained_model/model_epoch_15.pth  results/trained_model/model_epoch_15_16_4.pth
python3 main.py --pc 32 --nets 4 --primary-unit-size 8192 > f32_4.txt
mv     results/trained_model/model_epoch_15.pth  results/trained_model/model_epoch_15_32_4.pth

