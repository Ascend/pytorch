# Configuring pip Source

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T03:37:17.991Z pushedAt=2026-06-15T07:27:21.194Z -->

Configure the pip source as follows:

1. If you use pip to install packages, run the following command:

    ```bash
    cd ~/.pip
    ```

    If the directory does not exist, run the following command to create it:

    ```bash
    mkdir ~/.pip 
    cd ~/.pip
    ```

2. Edit the pip.conf file.

    Use the **vi pip.conf** command to open the pip.conf file, press the **i** key to enter edit mode, and write the following content:

    ```ini
    [global]
    #The following is an example, replace it with the actual repo.
    index-url = https://mirrors.huaweicloud.com/repository/pypi/simple
    trusted-host = mirrors.huaweicloud.com
    timeout = 120
    ```

3. Press the **Esc** key to exit edit mode, type **:wq!** and press Enter to save and exit the file.
