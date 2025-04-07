Setup Instructions

1. Install Python 3.9.0

    Download and install Python 3.9.0 from the official Python website.

    During installation, make sure to check the box to "Add Python to PATH".

2. Download FFmpeg

    In the dataset folder of this repository, there is a zip file named ffmpeg-master-latest-win64-gpl-shared.zip.

    Extract the contents of the zip file to a location of your choice.

3. Configure FFmpeg Path

    Inside the extracted folder, navigate to the bin directory.

    Copy the full path of the bin folder (e.g., C:\path\to\ffmpeg\bin).

4. Add FFmpeg to System PATH

    On Windows, open the Start menu and search for Environment Variables.

    Click on Edit the system environment variables.

    In the System Properties window, click Environment Variables.

    Under System variables, find and select the Path variable, then click Edit.

    In the Edit Environment Variable window, click New and paste the path to the bin directory you copied earlier (e.g., C:\path\to\ffmpeg\bin).

    Click OK to save your changes.

5. Install Dependencies

    Install the required Python libraries using pip. Run the following command in your terminal:

pip install -r requirements.txt
