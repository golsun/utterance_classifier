(this is a private repo for MSR AI NLP team)

This repo provides the code to extract "dialogs" from Reddit data, and score them on how "Twitter-like" they are. Twitter data is usually more causual while Reddit data usually contains more specific contents.
# Reddit data
As we don't own the Reddit data, we only provide the script here to process the data.

Steps:
* Download the raw data from a [third part](http://files.pushshift.io/reddit/comments/). The file names are in format `<YYYY-MM>`.
* Extract valid submissions (i.e. the "main" post) and their valid comments by `python reddit.py <YYYY-MM> --task=extract`
* Extract valid dialogs from these Reddit posts by `python reddit.py <YYYY-MM> --task=conv`

Here `<YYYY-MM>` stands for the file name you want to process. If there're multiples files you want to process, you can easily do so by a bash file by `sh run_all.sh`, where `run_all.sh` looks like this:
```
python reddit 2011-01 --task=extract
python reddit 2011-01 --task=conv
python reddit 2011-02 --task=extract
python reddit 2011-02 --task=conv
...
```
# Twitter/Reddit Classifier
With Reddit dialogs extracted above, you can run a trained classifier to score each dialog how "Twitter-like" they are. 

Steps
* Collect dialogs in a text file that each line is a tokenized dialog and has the format `context \t response`, if context has multiple turns, these turns should be delimited by `EOS`. For example `hello , how are you ? EOS not bad . how about yourself \t pretty good .`
* Score each dialog by `python classifier.py score --path=<path>`. This will generate a file `<path>.scored` that each line has the format `context \t response \t score`, where `score` is a number in the range from 0 to 1. 0 means very Reddit-like and 1 means Twitter-like

