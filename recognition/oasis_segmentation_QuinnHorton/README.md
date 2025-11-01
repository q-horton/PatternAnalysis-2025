# OASIS Brain Data Segmentation

***Implemented by Quinn Horton (46975919)***

<!-- This part is the description / info about the problem solved. -->
This model addresses the task of segmenting OASIS brain data.

## Dependencies

This has been developed using python version 3.13.9, and relies on the following other dependencies:
- os
- pytorch ('torch')
- torchvision
- dotenv

## Setup

Once all dependencies above have been installed, then begin by acquiring a local copy of the relevant OASIS dataset and setting up the relevant environment variables for the path. If running the program on rangpur, this can simply be achieved by navigating into the `oasis_segmentation_QuinnHorton` directory and running the following command:
```bash
echo "OASIS_PATH=/home/groups/comp3710/OASIS" >> .env
```

Otherwise, if this is being run on a local machine, first login to the rangpur server and create an archive of the dataset, which can be done with:
```bash
cd /home/groups/comp3710/OASIS
zip ~/.OASIS_DATA *
```

Then, exit the server and copy the archive onto the local machine using secure copy:
```bash
scp username@rangpur.labs.eait.uq.edu.au:~/OASIS_DATA.zip local-filepath
```

Finally, extract the contents of this archive into the desired location and create the required path environment variable:
```bash
echo "OASIS_PATH=/path/to/the/dataset" >> .env
```
