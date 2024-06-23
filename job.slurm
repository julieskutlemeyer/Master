#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --mem=250G
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name="Modelltrening"
#SBATCH --output=first_training.out


#før skrev jeg over: gres_gpu:3, --mem=120G, nodes=1, a100:1
# --mem-per-cpu=32G
#-time=01:00:00 var default over
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Directory: $SLURM_SUBMIT_DIR | Name of job: $SLURM_JOB_NAME | Job ID: $SLURM_JOB_ID"

nvidia-smi ---> bruk denne for å skjekke ut info om gpuen du bruker.

module purge
module load torchvision/0.13.1-foss-2022a-CUDA-11.7.0

#scancel -u juliemey
#pip install img2vec_pytorch -q
pip install einops -q
pip install deepface -q
pip install facenet-pytorch -q
pip install opencv-python -q
pip install numpy -q
pip install tqdm -q
pip install matplotlib -q
pip install transformers -q
pip install diffusers["torch"] transformers -q
pip install pandas -q
pip install IPython -q
pip install pytorch_pretrained_vit -q
pip install vit-pytorch -q
echo "Kjører python program"
python ddpm_OFFICIAL_resnet+label2.py ##ddpm_OFFICIAL_without_vae_img_embs.py
#sbatch job.slurm

#module load torchvision/0.13.1-foss-2022a-CUDA-11.7.0
##module load PyTorch/1.8.1-fosscuda-2020b
#module load torchvision/0.8.2-fosscuda-2020b-PyTorch-1.7.1
#pip install sentence-transformers -q
#pip install kaggle -q
#mkdir ~/.kaggle
#cp kaggle.json ~/.kaggle/
#chmod 600 ~/.kaggle/kaggle.json
#kaggle datasets download steubk/wikiart
#echo "download funket"
#unzip wikiart
#nvidia-smi



#for å kjøre denne: i terminalen: sbatch job.slurm
#denne kjører alt det over som shell kommandoer i idun-pcen, som skjønner sbatch osv. 

#du får ingen beskjed når ferdig, derfor trykke på refresh knappen ved Master til venstre i filsystemet

#for å ikke få feilmeldinger på ting så må du skrive pip3 install <pakken> i venven. grunnen er bare for å få forslag osv. i vscode, ikke fordi det 
#faktisk hade blitt fiel om du kjørte den i idun. 
#derfor gjør dette med numpy, matplotlib osv osv .



#slurm: api for å konfigurere jobs, snakker til idun. 
#job.slurm -> en fil som lager en job i idun
# du er jo koblet til idun nå remote right. og når du kjører denne så sender du api meldinger til idun via .slurm, som idun forstår

#CUDA er api til gpuer som er laget av nvidia. idun sine gpuer bruker cuda api for å koble seg på. derfor cuda_is_available

#for at torch.cuda apier skal fungere, må gpuen du ønsker å få "kontakt" med være cuda compatible. det er idun sine gpuer. 







#LMOD: api for å loade moduler. idun har node, pytohn alt dette i en liten lukket boks et spesielt sted på idun. når du skriver 
#module load fosscuda/2018b går du til adr på idun der alt dette finnes, og kommer inn i dette huset, og kan kjøre koden din der inne

#module spider -> søke etter pakker du kan laste ned
#module load er for å loade/laste ned
#module purge er for å slette alt
#module load torchvision/0.8.2-fosscuda-2020b-PyTorch-1.7.1 ----> denne inneholder alle pakker du trenger osv. 
#module list


#kjør denne i terminalen under her hvis du lager en ny job: chmod u+x <filnavn>.slurm


# bruk chatgpt og google slurm hvis du trenger noe mer hjelp.






### vit herfra: https://stackoverflow.com/questions/75874965/how-do-i-extract-features-from-a-torchvision-visitiontransfomer-vit


