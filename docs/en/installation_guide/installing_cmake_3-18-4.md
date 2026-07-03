# Installing CMake 3.18.4

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-15T03:37:49.780Z pushedAt=2026-06-15T07:27:21.203Z -->

The following steps should be performed by the **root** user.

1. In any directory on the server (for example, **/home**), execute the following command to obtain the cmake package.

    ```bash
    wget https://cmake.org/files/v3.18/cmake-3.18.4.tar.gz
    ```

2. Extract the package and enter its directory.

    ```bash
    tar -xf cmake-3.18.4.tar.gz
    cd cmake-3.18.4/
    ```

3. Execute the configuration, compilation, and installation commands.

    ```bash
    ./configure --prefix=/usr/local/cmake
    make && make install
    ```

4. Set up a symbolic link.

    ```bash
    ln -s /usr/local/cmake/bin/cmake /usr/bin/cmake
    ```

    If the above command reports the error "ln: failed to create symbolic link '/usr/bin/cmake': File exists", it indicates that a symbolic link ('/usr/bin/cmake') already exists. In this case, run the following command to delete the symbolic link (the symbolic link '/usr/bin/cmake' is only an example. Use the actual path on your system):

    ```bash
    rm -rf /usr/bin/cmake
    ```

    Then run the **ln -s /usr/local/cmake/bin/cmake /usr/bin/cmake** command to set up the symbolic link.

5. Run the following command to verify whether the installation is successful.

    ```bash
    cmake --version
    ```

    If "cmake version 3.18.4" is printed, the installation is successful.
