# timescale-diffusion

## Getting Started
This project uses the `uv` build system. you can find more about that at https://github.com/astral-sh/uv
This means that running an experiment is as easy as 
```sh
uv run $EXPERIMENT_NAME $ARGS
```
kind of.. the cluster doesn't actually have uv installed by default. 
instead you can just download uv w/ curl. while on the head-node, do:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```
then while you are in an interactive job you can do `source /cluster/home/$USER/.cargo/env` and you are off to the races.

### Jobs
if you don't know what an interactive job is, it the way in which you can access compute resources on the cluster
'interactively' that is you will get a shell instance on some compute node. this is in contrast to a normal job,
where you must define a script, usually `job.sh` in order to do anything. 
on the cluster, you can look at `/c/r-g/w/home/finn/timescale-diffusion` to see my job.sh, basically I set some 
environment variables, `source` my uv install, and then launch the program. things you may expect to work like
`$HOME` will not work in the job.sh script, so you must spell out the whole path to your `.cargo/env` file.
notably in the job.sh is the wandb api key, make sure to change this to yours other wise you will just send
runs to my wandb.

in order to actually launch a job look for the dispatch script. it should be called `dispatch.sh` or very similar.
in order to launch a non-interactive job you would do `./dispatch.sh $EXPERIMENT_NAME $ARGs`
and in order to launch an interactive job do `./dispatch.sh -i`. you can also change what resources you get when you
dispatch, for example
```sh
./dispatch.sh -g 2 -c 16 -m 64 -i
```
will give you an interactive session with 2 gpus, 16 cpus, and 64gb of ram
