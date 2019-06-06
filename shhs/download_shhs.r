## nsrr download; national sleep research resource

library(nsrr)
# nsrr_auth()
# Set the token by adding this to your ~/.Renviron file:
# NSRR_TOKEN="YOUR TOKEN GOES HERE"

path1='/ssd/hongyang/2018/physionet/data/shhs/'

## download polysomnogram edf
files = nsrr_dataset_files("shhs", path= "polysomnography/edfs/shhs1")
for (i in 1:nrow(files)){
    print(i)
    x=nsrr_download_file("shhs", path = files[i,2])
    system(paste0('mv ', x$outfile, ' ', path1, unlist(strsplit(files[i,2],'/'))[4]))
}

## download annotation xml
files = nsrr_dataset_files("shhs", path= "polysomnography/annotations-events-nsrr/shhs1")
for (i in 1:nrow(files)){
    print(i)
    x=nsrr_download_file("shhs", path = files[i,2])
    system(paste0('mv ', x$outfile, ' ', path1, unlist(strsplit(files[i,2],'/'))[4]))
}


## download polysomnogram edf
files = nsrr_dataset_files("shhs", path= "polysomnography/edfs/shhs2")
for (i in 1:nrow(files)){
    print(i)
    x=nsrr_download_file("shhs", path = files[i,2])
    system(paste0('mv ', x$outfile, ' ', path1, unlist(strsplit(files[i,2],'/'))[4]))
}

## download annotation xml
files = nsrr_dataset_files("shhs", path= "polysomnography/annotations-events-nsrr/shhs2")
for (i in 1:nrow(files)){
    print(i)
    x=nsrr_download_file("shhs", path = files[i,2])
    system(paste0('mv ', x$outfile, ' ', path1, unlist(strsplit(files[i,2],'/'))[4]))
}




