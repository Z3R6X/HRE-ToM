# **HRE-ToM** - **H**olistic **R**easoning **E**valuation for **T**heory-**o**f-**M**ind



## Data Generation
The directory 'DataGeneration contains the scripts to generate the scenarios of the Inferring Emotion and Knowledge-Perception task. 

- Activate the conda environment:
    ```
    conda activate datagen
    ```

- To generate the scenarios for the Inferring Emotion task run:
    ```
    python DataGeneration/inferring_emotion/generate_scenarios.py --emotion <target emotion for the scenario> [options]
    ```
    Arguments:
    - ```--num_samples```: Number of samples generated per basic scenario
    - ```--basic_scenarios```: JSON file with basic scenarios used to generate the full scenarios, default: ```basic_scenarios.json```
    - ```--names```: JSON file with common names used to generate the full scenarios, default: ```common_names.json```

    

- To generate the scenarios for the Knowledge-Perception task run:
    ```
    python DataGeneration/knowledge-perception/generate_scenarios.py 
    ```
    Arguments:
    - ```--num_samples```: Number of samples generated per basic scenario
    - ```--basic_scenarios```: JSON file with basic scenarios used to generate the full scenarios, default: ```basic_scenarios.json```
    - ```--names```: JSON file with common names used to generate the full scenarios, default: ```common_names.json```


