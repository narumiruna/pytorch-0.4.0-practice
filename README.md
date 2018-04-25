# pytorch-0.4.0-practice

## Build and run
```
$ docker build -t pytorch .
$ nvidia-docker run -it --rm -v $(pwd):/workspace pytorch python3 main.py
```