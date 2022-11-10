if [[ $1 = "all" ]];
then
    echo "Installing all packages, including Airsim and UE4"
    airsim=1
    ue4=1
elif [[ $1 = "airsim" ]];
then
    echo "Installing repo and Airsim"
    airsim=1
    ue4=0
elif [[ $1 = "ue4" ]];
then
    echo "Installing repo and UE4"
    airsim=0
    ue4=1
else
  echo "Installing base packages"
    airsim=0
    ue4=0

conda create -n llp_mpc python=3.8 -y
eval "$(conda shell.bash hook)"
conda activate llp_mpc
pip install -e .
pip install -r requirements.txt

git clone https://github.com/abr/abr_control
cd abr_control
pip install -e .
git fetch
git checkout b42c193779983f6d6f8da5403a5593f8f0ff50b9
cd ..

git clone https://github.com/abr/abr_analyze
cd abr_analyze
git fetch
git checkout 6a1bf968055f861843abfa44fdc60af7e9c4b399
python setup.py install
cd ..


fi
if [[ $airsim == 1 ]];
then
    echo "Installing airsim, this may take a few minutes"
    git clone git@github.com:p3jawors/AirSim.git
    cd AirSim
    git checkout payload
    ./setup.sh
    ./build.sh
    cd PythonClient
    pip install -e .
    cd ../../
fi

if [[ $ue4 == 1 ]];
then
    echo "Installing UE4, this may takes upwards of a few hours..."
    git clone -b 4.26 git@github.com:EpicGames/UnrealEngine.git
    cd UnrealEngine
    git checkout tags/4.26.0-release
    ./Setup.sh
    ./GenerateProjectFiles.sh
    make
    cd ..
fi


