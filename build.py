#!/usr/bin/env python3
###############################################################################
## MIT License
##
## Copyright (c) 2022 Alain Duron <duron.alain@gmail.com>
## Copyright (c) 2022 Quentin Quadrat <lecrapouille@gmail.com>
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
###############################################################################
###
### This python script allows to compile CEF helloworld project for Linux or
### Windows.
###
###############################################################################

import os, sys, subprocess, hashlib, tarfile, shutil, glob, progressbar, urllib.request, shutil
from platform import machine, system
from pathlib import Path
from subprocess import run
from multiprocessing import cpu_count
from packaging import version

###############################################################################
### Global user settings
BUILD_TYPE = "DEBUG" #or RELEASE
MODULE_TARGET = "template_debug" if BUILD_TYPE == "DEBUG" else "template_release"          # or "template_debug"
GODOT_CPP_TARGET = "template_debug" if BUILD_TYPE == "DEBUG" else "template_release" 
GODOT_VERSION = "4.0"              # or "3.4"
CUDA_VERSION = "11.4" # or "11.8" for Orin
CUDA_ARCH_BIN = "8.6"



PWD = os.getcwd()
EYE_BUILD_PATH = PWD#os.path.join(PWD, "gd-eye-I-develop")
GDEYEI_THIRDPARTY_PATH = os.path.join(PWD, "thirdparty")
THIRDPARTY_GODOT_PATH = os.path.join(GDEYEI_THIRDPARTY_PATH, "godot-" + GODOT_VERSION)
GODOT_CPP_API_PATH = os.path.join(THIRDPARTY_GODOT_PATH, "cpp")
OPENCV_CPP_API_PATH = os.path.join(GDEYEI_THIRDPARTY_PATH, "opencv")
OPENCV_CPP_BUILD_PATH = os.path.join(OPENCV_CPP_API_PATH, "build")
OPENCV_CPP_INSTALL_PATH = os.path.join(OPENCV_CPP_API_PATH, "install")
OPENCV_CONTRIB_CPP_API_PATH = os.path.join(GDEYEI_THIRDPARTY_PATH, "opencv_contrib")
OPENCV_CONTRIB_MODULES_PATH = os.path.join(OPENCV_CONTRIB_CPP_API_PATH, "modules")
DLIB_CPP_API_PATH = os.path.join(GDEYEI_THIRDPARTY_PATH, "dlib")
GODOT_CPP_API_FILE = os.path.join(GODOT_CPP_API_PATH, "gdextension/extension_api.json")


OPENCV_CONFIGURE_DICTIONARY = {
    "CMAKE_INSTALL_PREFIX" : OPENCV_CPP_INSTALL_PATH,
    "CMAKE_BUILD_TYPE" : BUILD_TYPE,
	"INSTALL_PYTHON_EXAMPLES" : "OFF",
	"WITH_TBB" : "ON",
	"OPENCV_ENABLE_NONFREE" : "ON",
	"WITH_CUDA" : "ON",
	"WITH_CUDNN" : "ON",
	"OPENCV_DNN_CUDA" : "ON",
	"ENABLE_FAST_MATH" : "1",
	"CUDA_FAST_MATH" : "1",
	"CUDA_ARCH_BIN" : CUDA_ARCH_BIN,
	"WITH_CUBLAS" : "1",
	"WITH_OPENGL" : "OFF",
	"WITH_QT" : "OFF",
	"OPENCV_EXTRA_MODULES_PATH" : OPENCV_CONTRIB_MODULES_PATH,
	"BUILD_EXAMPLES" : "OFF",
    "BUILD_SHARED_LIBS" : "ON",
    "BUILD_STATIC_LIBS" : "OFF",
    "BUILD_opencv_python2" : "OFF", 
    "BUILD_opencv_python3" : "OFF"
}

###############################################################################
### Type of operating system, AMD64, ARM64 ...
ARCHI = machine()
NPROC = str(cpu_count())
OSTYPE = system()

###############################################################################
### Green color message
def info(msg):
    print("\033[32m[INFO] " + msg + "\033[00m", flush=True)

###############################################################################
### Red color message + abort
def fatal(msg):
    print("\033[31m[FATAL] " + msg + "\033[00m", flush=True)
    sys.exit(2)

###############################################################################
### Equivalent to test -L e on alias + ln -s
def symlink(src, dst, force=False):
    p = Path(dst)
    if p.is_symlink():
        os.remove(p)
    elif force and p.is_file():
        os.remove(p)
    elif force and p.is_dir():
        rmdir(dst)
    os.symlink(src, dst)

###############################################################################
### Equivalent to cp --verbose
def copyfile(file_name, folder):
    dest = os.path.join(folder, os.path.basename(file_name))
    print("Copy " + file_name + " => " + dest)
    shutil.copyfile(file_name, dest)

###############################################################################
### Equivalent to mkdir -p
def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

###############################################################################
### Equivalent to rm -fr
def rmdir(top):
    if os.path.isdir(top):
        for root, dirs, files in os.walk(top, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(top)

###############################################################################
### Equivalent to tar -xj
def untarbz2(tar_bz2_file_name, dest_dir):
    info("Unpacking " + tar_bz2_file_name + " ...")
    with tarfile.open(tar_bz2_file_name) as f:
        directories = []
        root_dir = ""
        for tarinfo in f:
            if tarinfo.isdir() and root_dir == "":
                root_dir = tarinfo.name
            name = tarinfo.name.replace(root_dir, dest_dir)
            print(" - %s" % name)
            if tarinfo.isdir():
                mkdir(name)
                continue
            tarinfo.name = name
            f.extract(tarinfo, "")

###############################################################################
### Search an expression (not a regexp) inside a file
def grep(file_name, what):
    try:
        file = open(file_name, "r")
        for line in file:
            if line.find(what) != -1:
                return line
        return None
    except IOError:
        return None

###############################################################################
### Needed for urllib.request.urlretrieve
### See https://stackoverflow.com/a/53643011/8877076
class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()

###############################################################################
### Download artifacts
def download(url, destination):
    info("Download " + url + " into " + destination)
    urllib.request.urlretrieve(url, destination, reporthook=MyProgressBar())
    print('', flush=True)

###############################################################################
### Compute the SHA1 of the given artifact file
def compute_sha1(artifact):
    CHUNK = 1 * 1024 * 1024
    sha1 = hashlib.sha1()
    with open(artifact, 'rb') as f:
        while True:
            data = f.read(CHUNK)
            if not data:
                break
            sha1.update(data)
    return "{0}".format(sha1.hexdigest())

###############################################################################
### Read a text file holding a SHA1 value
def read_sha1_file(path_sha1):
    file = open(path_sha1, "r")
    for line in file:
        return line # Just read 1st line
    return None

###############################################################################


def download_godot_cpp():
    if not os.path.exists(GODOT_CPP_API_PATH):
        info("Clone cpp wrapper for Godot " + GODOT_VERSION + " into " + GODOT_CPP_API_PATH)
        mkdir(GODOT_CPP_API_PATH)
        run(["git", "clone", "--recursive", "-b", GODOT_VERSION,
             "https://github.com/godotengine/godot-cpp", GODOT_CPP_API_PATH])

def download_opencv_cpp():
    if not os.path.exists(OPENCV_CPP_API_PATH):
        info("Clone opencv library into " + OPENCV_CPP_API_PATH)
        mkdir(OPENCV_CPP_API_PATH)
        run(["git", "clone", 
             "https://github.com/opencv/opencv.git", OPENCV_CPP_API_PATH])
        
def download_opencv_contrib_cpp():
    if not os.path.exists(OPENCV_CONTRIB_CPP_API_PATH):
        info("Clone opencv library into " + OPENCV_CONTRIB_CPP_API_PATH)
        mkdir(OPENCV_CONTRIB_CPP_API_PATH)
        run(["git", "clone", 
             "https://github.com/opencv/opencv_contrib.git", OPENCV_CONTRIB_CPP_API_PATH])
        
def download_dlib_cpp():
    if not os.path.exists(DLIB_CPP_API_PATH):
        info("Clone opencv library into " + DLIB_CPP_API_PATH)
        mkdir(DLIB_CPP_API_PATH)
        run(["git", "clone", 
             "https://github.com/davisking/dlib.git", DLIB_CPP_API_PATH])


###############################################################################
### Compile Godot cpp wrapper needed for our gdnative code: CEF ...
def compile_godot_cpp():
    lib = os.path.join(GODOT_CPP_API_PATH, "bin", "libgodot-cpp*" + GODOT_CPP_TARGET + "*")
    if not os.path.exists(lib):
        info("Compiling Godot C++ API (inside " + GODOT_CPP_API_PATH + ") ...")
        os.chdir(GODOT_CPP_API_PATH)
        if OSTYPE == "Linux":
            run(["scons", "platform=linux", "target=" + GODOT_CPP_TARGET,
                 "--jobs=" + NPROC], check=True)
        elif OSTYPE == "Darwin":
            run(["scons", "platform=osx", "macos_arch=" + ARCHI,
                 "target=" + GODOT_CPP_TARGET, "--jobs=" + NPROC], check=True)
        elif OSTYPE == "MinGW":
            run(["scons", "platform=windows", "use_mingw=True",
                 "target=" + GODOT_CPP_TARGET, "--jobs=" + NPROC], check=True)
        elif OSTYPE == "Windows":
            run(["scons", "platform=windows", "target=" + GODOT_CPP_TARGET,
                 "--jobs=" + NPROC], check=True)
        else:
            fatal("Unknown architecture " + OSTYPE + ": I dunno how to compile Godot-cpp")

def add_path_to_env_variable(var_name, path):
    old_var = os.environ.get(var_name, "")
    new_var = path
    if old_var != "":
        new_var += ":" + old_var
    os.environ[var_name] = new_var


def compile_opencv():
    if (os.path.exists(OPENCV_CPP_INSTALL_PATH)):
        return
    add_path_to_env_variable("LD_LIBRARY_PATH", "/usr/local/cuda-" + CUDA_VERSION + "/lib64")
    add_path_to_env_variable("PATH", "/usr/local/cuda-" + CUDA_VERSION + "/bin")
    info("Configuring OpenCV...")

    os.chdir(OPENCV_CPP_API_PATH)
    mkdir(OPENCV_CPP_BUILD_PATH)
    os.chdir(OPENCV_CPP_BUILD_PATH)
    conf_arr = [f'{key}={value}' for (key, value) in OPENCV_CONFIGURE_DICTIONARY.items()]
    conf_arr_new = []
    for arg in conf_arr:
        conf_arr_new.append("-D")
        conf_arr_new.append(arg)
    args = ["cmake" , ".."] + conf_arr_new
    run(args)
    info("Build OpenCV")
    run(["make", "-j" + NPROC])
    run(["make", "install"])
    #shutil.rmtree(OPENCV_CPP_BUILD_PATH)

###############################################################################
### Check if compilers are present (Windows)
def check_compiler():
    if OSTYPE == "Windows":
        cppfile = "win.cc"
        binfile = "win.exe"
        objfile = "win.obj"
        with open(cppfile, "w") as f:
            f.write("#include <windows.h>\n")
            f.write("int main(int argc, char **argv) { return 0; }")
        if os.system("cl.exe /Fe:" + binfile + " " + cppfile) != 0:
            os.remove(cppfile)
            fatal("MS C++ compiler is not found. "
                  "Install https://visualstudio.microsoft.com "
                  "and open an x64 Native Tools Command Prompt for VS 2022, with Administrator privilege")
        if os.path.isfile(binfile) == False:
            os.remove(cppfile)
            fatal("MS C++ compiler is not working. "
                  "Install https://visualstudio.microsoft.com "
                  "and open an x64 Native Tools Command Prompt for VS 2022, with Administrator privilege")
        if os.system(binfile) != 0:
            os.remove(cppfile)
            fatal("MS C++ compiler could not compile test program. "
                  "Install https://visualstudio.microsoft.com "
                  "and open an x64 Native Tools Command Prompt for VS 2022, with Administrator privilege")
        info("MS C++ Compiler OK")
        os.remove(cppfile)
        os.remove(binfile)
        os.remove(objfile)

###############################################################################
### Common Scons command for compiling our Godot gdnative modules
def gdnative_scons_cmd(platform):
    if GODOT_CPP_API_PATH == '':
        fatal('Please download and compile https://github.com/godotengine/godot-cpp and set GODOT_CPP_API_PATH')
    if platform == "macos":
        run(["scons", "api_path=" + GODOT_CPP_API_PATH,
             "target=" + MODULE_TARGET, "--jobs=" + NPROC,
             "arch=" + ARCHI, "godot_version=" + GODOT_VERSION,"platform=" + platform], check=True)
    else:
        run(["scons", "api_path=" + GODOT_CPP_API_PATH,
             "custom_api_file=" + GODOT_CPP_API_FILE, "godot_version=" + GODOT_VERSION,
             "target=" + MODULE_TARGET, "--jobs=" + NPROC, "arch=" + ARCHI,
             "platform=" + platform], check=True)

###############################################################################
### Compile extension module
def compile_extension(path):
    info("Compiling extension module " + path)
    os.chdir(path)
    if OSTYPE == "Linux":
        gdnative_scons_cmd("linux")
    elif OSTYPE == "Darwin":
        gdnative_scons_cmd("macos")
    elif OSTYPE == "Windows": # or OSTYPE == "MinGW":
        gdnative_scons_cmd("windows")
    else:
        fatal("Unknown archi " + OSTYPE + ": I dunno how to compile module primary process")

def copy_extension_to_path(path):
    src_files = os.listdir(os.path.join(EYE_BUILD_PATH, "addon/bin"))
    dst_dir = os.path.join(path, "bin")
    if not os.path.exists(dst_dir):
        mkdir(dst_dir)
    for file_name in src_files:
        full_file_name = os.path.join(os.path.join(EYE_BUILD_PATH, "addon/bin"), file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dst_dir)
    shutil.copy(os.path.join(EYE_BUILD_PATH, "addon/eye-recognizer.gdextension"), path)
    pass

###############################################################################
### Entry point
if __name__ == "__main__":
    check_compiler()
    download_godot_cpp()
    download_opencv_cpp()
    download_opencv_contrib_cpp()
    download_dlib_cpp()
    compile_opencv()
    compile_godot_cpp()
    compile_extension(EYE_BUILD_PATH)
    if (len(sys.argv) > 1):
        copy_extension_to_path(sys.argv[1])

