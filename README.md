<div style="color: red">※ 2018/1/6時点</div>
<div style="color: red">※ GPUが積んである前提で書いています</div>

* [最初の設定](#setting)
* [CUDA toolkitのインストール](#cuda)
* [Docker CE + NVIDIA Dockerのインストール](#docker)
* [Optionalな最初の設定](#optional_setting)
* [Git周りの設定](#git)

「あとはDockerでやるぜ」って人はここから先は見る必要ないです。好きにしてください。
* [cuDNNのインストール](#cudnn)
* [pyenv + AnacondaでPython3.6のインストール](#python)
* [OpenCV3.3 + contribのインストール](#opencv)
* [Tensorflow + Kerasのインストール](#keras)
* [Boostのインストール](#boost)

<!------------------------------------------------------------------------------------------------>
# <a name="setting">最初の設定</a>

## なにはともあれ
```sh
sudo apt update
sudo apt -y upgrade
```

もし日本語RemixからUbuntuをインストールしたときは以下も必要。
## システムの言語を英語→日本語に変更
```sh
LANG=C xdg-user-dirs-gtk-update
```

<!------------------------------------------------------------------------------------------------>
# <a name="cuda">CUDA toolkitのインストール（要、再起動）</a>
CUDA toolkitを入れることで、最新版のGraphic Driverもいっしょに入ります。

ということで、まずは https://developer.nvidia.com/cuda-downloads で以下を選択してインストーラー（あればパッチファイルも）をダウンロード
* Operating System -> Linux
* Architecture -> x86_64
* Distribution -> Ubuntu
* Version -> 16.04
* Installer Type -> deb (local)

※ CUDAのバージョンによってインストレーションが違ったりするので、ここの情報よりもダウンロード時に表示されるインストレーションに従うこと！
```sh
cd ~/Downloads
sudo dpkg -i cuda-repo-ubuntu1604-9-1-local_9.1.85-1_amd64.deb
sudo apt-key add /var/cuda-repo-9-1-local/7fa2af80.pub
sudo apt update
sudo apt install -y cuda
sudo reboot
```

`nvidia-smi` コマンドで以下のような表示が出れば成功
```sh
$ nvidia-smi
Sat Jan  6 21:45:21 2018
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 387.34                 Driver Version: 387.34                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1080    Off  | 00000000:03:00.0 Off |                  N/A |
|  0%   47C    P8    11W / 240W |      2MiB /  8114MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 1080    Off  | 00000000:04:00.0  On |                  N/A |
|  0%   50C    P8    17W / 240W |    922MiB /  8105MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

<!------------------------------------------------------------------------------------------------>
# <a name="docker">Docker CE + NVIDIA Dockerのインストール</a>
（参考）
* [（公式）Get Docker CE for Ubuntu](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#install-using-the-repository)
* [（公式）NVIDIA Docker Installation (version 2.0)](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0))

## Docker CE
```sh
sudo apt install -y apt-transport-https \
                    ca-certificates \
                    curl \
                    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
sudo apt install -y docker-ce
```

`sudo docker run --rm hello-world` で以下のような表示が出れば成功
```sh
$ sudo docker run --rm hello-world
Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://cloud.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/engine/userguide/
```

## NVIDIA Docker
```sh
curl -sL https://nvidia.github.io/nvidia-docker/gpgkey | \
         sudo apt-key add -
curl -sL https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list | \
         sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
```

`sudo docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi` で以下のような表示が出れば成功
```sh
$ sudo docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
Sat Jan  6 12:46:03 2018
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 387.34                 Driver Version: 387.34                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1080    Off  | 00000000:03:00.0 Off |                  N/A |
|  0%   47C    P8    11W / 240W |      2MiB /  8114MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 1080    Off  | 00000000:04:00.0  On |                  N/A |
|  0%   50C    P8    13W / 240W |    922MiB /  8105MiB |      1%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

## sudo なしでDockerを使えるようにする
毎回毎回 `sudo` をつけるのは面倒なので、`sudo` なしで `docker`, `nvidia-docker` コマンドを使えるようにする
```sh
sudo usermod -aG docker $USER
```

<!------------------------------------------------------------------------------------------------>
# <a name="optional_setting">Optionalな最初の設定</a>

## デフォルトのシェルをbash→zsh+oh-my-zshに変更
（参考）
[Ubuntuで開発環境構築その１：ターミナル（Zsh, oh-my-zsh, byobu, etc.）](http://snowsunny.hatenablog.com/entry/2014/04/03/211105)

zshはbashのパワーアップver.と思ってもらえればいいです。便利！
### 1. zshのインストール 
```sh
sudo apt install -y zsh
```
次に、
```sh
which zsh
```
でインストールしたzshの場所を確認して、`chsh` コマンドでその場所を入力する。
```shell
$ chsh
Password:
Changing the login shell for snhryt
Enter the new value, or press ENTER for the default
	Login Shell [/bin/bash]:** HERE **
```
入力後ターミナルを再起動すると、デフォルトのシェルがzshに変わります。

### 2. oh-my-zshのインストール
```sh
sudo apt install -y curl
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
```
これでだいぶ見易いシェルになったはず。
https://github.com/robbyrussell/oh-my-zsh/wiki/Themes にテーマが色々乗ってるので、~/.zshrc をとりあえず
```sh
ZSH_THEME="amuse"
```
のようにいじって自分好みにしていきましょう。

## （個人的）必須アプリのインストール
* Guake Terminal
* OpenSSH
* [Chrome](https://www.google.co.jp/chrome/browser/desktop/)
* [Visual Studio Code](https://code.visualstudio.com/download)

がないと個人的には何も始まらないのでこいつらを入れる。
ChromeとVSCodeはマニュアルで.debファイルをダウンロードしておく。
```sh
# Guake TerminalとOpenSSHのインストール
sudo apt install -y guake openssh-server
# Chromeのインストール
sudo apt install -y libappindicator1
sudo apt update
cd ~/Downloads
sudo dpkg -i google-chrome-stable_current_amd64.deb
# VScodeのインストール
sudo dpkg -i code_1.15.1-1502903936_amd64.deb
```

## ランチャーの位置を左→下に移動
Windowsに慣れてる人はこの設定のほうがしっくりくるかも。
完全に好みですね。
```sh
gsettings set com.canonical.Unity.Launcher launcher-position Bottom
```

## Mozcの設定
```sh
sudo apt install -y ibus-mozc
killall ibus-daemon
ibus-daemon -d -x &
```

## ファイラーからGoogle Driveにアクセス
[Ubuntu 16.04の標準ファイラーで「Google Drive」にアクセスする方法](http://ottan.xyz/ubuntu-16-04-google-drive-filer-4725/)

## （個人的）準必須アプリのインストール
必須ではないけど欲しいやつらとしては
* VLC media player（動画再生）
* Geeqie（画像ビューワー）
* Ubuntuソフトウェアセンター（Ubuntu Softwareは使い勝手が悪いので）
* CompizConfig Settings Manager（設定から変えられない部分をいじる）
* Unity Tweak Tool（設定から変えられない部分をいじる）

あたりでしょうか。全部 `apt install` で一撃です。
```sh
sudo apt install -y vlc \
                    geeqie \
                    software-center \
                    compizconfig-settings-manager \
                    unity-tweak-tool
```

## Amazonアプリのアンインストール
要らないでしょ？
```sh
sudo apt remove -y unity-webapps-common
```

<!--
## Spotifyのインストール
```sh
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys BBEBDCB318AD50EC6865090613B00F1FD2C19886
echo deb http://repository.spotify.com stable non-free | sudo tee /etc/apt/sources.list.d/spotify.list
sudo apt update
sudo apt install spotify-client
```
-->

<!------------------------------------------------------------------------------------------------>
# <a name="git">Git周りの設定</a>
## Gitの最新版をインストール
```sh
sudo add-apt-repository ppa:git-core/ppa
sudo apt update
sudo apt install -y git
```

## Gitの初期設定
[最初のGitの構成](https://git-scm.com/book/ja/v1/%E4%BD%BF%E3%81%84%E5%A7%8B%E3%82%81%E3%82%8B-%E6%9C%80%E5%88%9D%E3%81%AEGit%E3%81%AE%E6%A7%8B%E6%88%90)

[Gitをインストールしたら真っ先にやっておくべき初期設定](https://qiita.com/wnoguchi/items/f7358a227dfe2640cce3)

## SSHでGitに接続する
[gitHubでssh接続する手順~公開鍵・秘密鍵の生成から~](https://qiita.com/shizuma/items/2b2f873a0034839e47ce)

# <span style="color: red">=== 以降はDocker使わない人向けです ====</span>

<!------------------------------------------------------------------------------------------------>
# <a name="cudnn">cuDNNのインストール</a>
これがあるのとないので学習時間が10倍くらい変わってくるので、GPUを使うなら必需品。
https://developer.nvidia.com/cudnn から、CUDAに対応するバージョンをダウンロードしてくる。
なお、ダウンロードするためにはNVIDIA Developerの会員になる必要がある。
```sh
cd Downloads/
tar -zxvf cudnn-8.0-linux-x64-v7.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h \
               /usr/local/cuda/lib64/libcudnn*
```

<!------------------------------------------------------------------------------------------------>
# <a name="python">pyenv + AnacondaでPython3.6のインストール</a>
（参考）
* [【随時更新】pyenv + Anaconda (Ubuntu 16.04 LTS) で機械学習のPython開発環境をオールインワンで整える](http://blog.algolab.jp/post/2016/08/21/pyenv-anaconda-ubuntu/)
* [Anaconda を利用した Python のインストール (Ubuntu Linux)](http://pythondatascience.plavox.info/python%E3%81%AE%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB/anaconda-ubuntu-linux)

まずは必要なパッケージのインストールとpyenv周りの環境整備。（pyenvはAnacondaを入れるためだけに使う）
```sh
sudo apt install -y make \
                    build-essential \
                    libssl-dev \
                    zlib1g-dev \
                    libbz2-dev \
                    libreadline-dev \
                    libsqlite3-dev \
                    wget \
                    curl \
                    llvm \
                    libncurses5-dev \
                    libncursesw5-dev \
                    libpng-dev
git clone git://github.com/yyuu/pyenv.git ~/.pyenv
git clone https://github.com/yyuu/pyenv-pip-rehash.git ~/.pyenv/plugins/pyenv-pip-rehash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
```

pyenvからインストール可能なPython3系のAnacondaのリストを表示して、最新のバージョンを確認。
```sh
pyenv install -l | grep anaconda3
```

最新のバージョン（ここでは5.0.1）をインストールし、デフォルトとして設定する。
```sh
pyenv install anaconda3-5.0.1
pyenv global anaconda3-5.0.1
echo 'export PATH="$PYENV_ROOT/versions/anaconda3-5.0.1/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

最後にPythonのバージョンを確認して、以下のようになっていれば成功。
```sh
$ python -V
Python 3.6.3 :: Anaconda custom (64-bit)
```

以降は必要なパッケージがあれば、基本は `conda` 経由で、それが難しければ `pip` からインストールする。
ってことで、とりあえず両方ともアップデートしておく。
```sh
conda update -y conda
pip install -U pip
```

<!------------------------------------------------------------------------------------------------>
# <a name="opencv">OpenCV3.3 + contribのインストール</a>
あなたはC++（など）でOpenCVを使う予定がありますか？ [YES/NO]

## NOの人
```sh
pip install opencv-contrib-python
sudo apt install -y python-qt4
```
おしまい。 `pip` っょぃ。
samples/OpencvSample.py がエラーなく実行できればOK。

## YESの人
<span style="color: red">OpenCVは3.1(3.2？)と3.3でcmakeの引数名が変わっているので、旧バージョンのインストール方法だけを見ながら進めると必ず躓きます</span>
<!-- 
(参考)
* [まっさらなUbuntu16.04LTSにOpenCV3.1とopencv_contribをインストール](http://qiita.com/schwalbe1996/items/cfa9fde7ea2b825e4e6b)
* [Compile OpenCV 3.2 for Anaconda Python 3.6, 3.5, 3.4 and 2.7](https://www.scivision.co/anaconda-python-opencv3/)
* [（Github公式） OpenCV 3.2 Installation Guide on Ubuntu 16.04](https://github.com/BVLC/caffe/wiki/OpenCV-3.2-Installation-Guide-on-Ubuntu-16.04)
* [（Github issue） import xgboost OSError:version `GOMP_4.0' not found](https://github.com/dmlc/xgboost/issues/1786)
* [UbuntuにOpenCV3.2とcontribをインストールする。](http://shibafu3.hatenablog.com/entry/2017/03/28/164125)
* [OpenCV3.0とopencv_contribをubuntuに入れた作業メモ](http://qiita.com/iwata-n@github/items/6ae7e430e8008be29377)
* [Python3+OpenCV3.1＋ffmpegで動画を読み込む（Ubuntu16.04）](http://qiita.com/PonDad/items/cbef5dca04a1c1a201b0)
* [Ubuntu 16.04: How to install OpenCV](http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)
* [OpenCV Anaconda installation in Ubuntu](http://machinelearninguru.com/computer_vision/installation/opencv/opencv.html)
-->

### 1. ソースのダウンロード
公式のGitは重いので、別のリポジトリからcloneしてくる。
```sh
cd ~
git clone https://github.com/Itseez/opencv
git clone https://github.com/Itseez/opencv_contrib.git
```

### 2. ビルドに必要なパッケージのインストール
準備として、まずは /etc/apt/sources.list 中の
```txt
# deb-src http://jp.archive.ubuntu.com/ubuntu/ xenial universe`
```
の行を `sudo vi` とか `sudo nano` とかでアンコメントする。それから、
```sh
sudo apt update
sudo apt build-dep -y opencv
sudo apt install -y nvidia-opencl-dev
```

### 3. make
次に、~/tmp/opencv/CMakeLists.txt の先頭に
```cmake
set(CMAKE_CXX_FLAGS "-D_FORCE_INLINES ${CMAKE_CXX_FLAGS}")
```
を追加して、OpenCVを`make`する。
 `make` と `make install` はむちゃくちゃ時間がかかる（Core-i7 & 16コアの比較的いいマシンで合計1時間以上）ので、気長に待つ。
```sh
cd ~/opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules/ \
      -D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_python3=ON \
      -D PYTHON3_EXECUTABLE=$(which python3) \
      -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
      -D WITH_CUBLAS=ON \
      -D WITH_TBB=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D BUILD_EXAMPLES=ON \
      ..
make -j $(($(nproc) + 1))
make install -j $(($(nproc) + 1))
sudo /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig
sudo apt update
```

### 4. Pythonで使えるかのテスト
この状態でPythonを起動して `import cv2` とするとエラーが出るので、
```sh
# "`GLIBCXX_3.4.21' not found" のエラーに対応
cd ~/.pyenv/versions/anaconda3-4.4.0/lib
rm libstdc++.so.6
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6
# "`GOMP_4.0' not found" のエラーに対応
conda install libgcc
```
<!-- ```python
ImportError: /home/snhryt/.pyenv/versions/anaconda3-4.4.0/lib/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by /home/snhryt/.pyenv/versions/anaconda3-4.4.0/lib/python3.6/site-packages/cv2.cpython-36m-x86_64-linux-gnu.so)
```
```python
ImportError: /home/snhryt/.pyenv/versions/anaconda3-4.4.0/lib/libgomp.so.1: version `GOMP_4.0' not found (required by /usr/lib/x86_64-linux-gnu/libsoxr.so.0)
``` -->
最後に samples/OpencvSample.py と samples/OpencvSample_cpp/main.cpp がエラーなく実行できればOK。


<!------------------------------------------------------------------------------------------------>
# <a name="keras">Tensorflow + Kerasのインストール</a>
（参考）
* [Installing TensorFlow on Ubuntu](https://www.tensorflow.org/install/install_linux)
* [Keras公式](https://keras.io/ja/#_2)
* [Condaを使ってTensorFlowの野良ビルドを導入する](http://louis-needless.hatenablog.com/entry/try-tensorflow-with-anaconda)
* [【随時更新】pyenv + Anaconda (Ubuntu 16.04 LTS) で機械学習のPython開発環境をオールインワンで整える](http://blog.algolab.jp/post/2016/08/21/pyenv-anaconda-ubuntu/)
* [KerasでMNIST | 人工知能に関する断創録](http://aidiary.hatenablog.com/entry/20161109/1478696865)

まずはTensorFlowのインストール。
Anaconda経由でGPU & Python3.6対応のTensorFlowをインストールするので、[ここ](https://www.tensorflow.org/install/install_linux#InstallingAnaconda)と[ここ](https://www.tensorflow.org/install/install_linux#the_url_of_the_tensorflow_python_package)を参考にインストール。
（`conda install` でもいけるみたいだけど、 `pip` から入れるほうがいいみたい）
```sh
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp36-cp36m-linux_x86_64.whl
```

Kerasは
```sh
pip install keras
```
で簡単にインストールできる。
あとは KerasSample.py がエラーなく実行できればOK。

<!------------------------------------------------------------------------------------------------>
# <a name="boost">Boostのインストール</a>
## マニュアルで入れる方法
（参考）
* [（公式）boost C++ LIBRARIES](http://www.boost.org/users/download/#live)
* [Boostライブラリのビルド方法](https://boostjp.github.io/howtobuild.html)

こっちの方法がオススメ。
「絶対に最新版！」というこだわりがない場合は、最初のダウンロード部分は

`git clone --recursive git@github.com:boostorg/boost.git` でもOK
```sh
cd ~
wget https://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.gz
tar xfz boost_1_65_1.tar.gz
rm boost_1_65_1.tar.gz
cd boost_1_65_1
./bootstrap.sh
./b2 --without-python --prefix=/usr -j $(($(nproc) + 1)) link=shared runtime-link=shared install
cd ..
rm -rf boost_1_65_1
sudo ldconfig
```

## apt-getから入れる方法
（参考） [How to Install boost on Ubuntu?](https://stackoverflow.com/questions/12578499/how-to-install-boost-on-ubuntu)

多くの場合、このやり方だと後々面倒になる…
```sh
sudo apt install -qqy libboost-all-dev
```
