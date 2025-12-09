## Data folder
To make the code not show local paths, the direcotry containing the data was
linked with a symlink to a relative path

```sh
mkdir example_data
ln -s /mnt/lacus/example_soil_harmo/* example_data/
```