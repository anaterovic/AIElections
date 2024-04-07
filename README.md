# Git tutorial for pushing the changes directly to a branch 

### Optional: Installing Git on Windows

1. **Download Git for Windows**: Visit [https://gitforwindows.org/](https://gitforwindows.org/) and download the Git installer.

2. **Install Git**: Run the downloaded installer. During installation, you can leave the default settings, or customize as per your preferences.

3. **Verify Installation**: Open your Command Prompt (cmd) or Git Bash and type `git --version` to ensure Git was installed successfully.

### Cloning a Repository and Working with Branches

1. **Open Git Bash**: You can do this by right-clicking in any folder and selecting "Git Bash Here".

2. **Clone the Repository**: Use the `git clone` command followed by the URL of the repository. 
   ```
   git clone <repository-url>
   ```
   Replace `<repository-url>` with the actual URL of the repository.

3. **Navigate to the Repository Directory**: 
   ```
   cd <repository-name>
   ```
   Replace `<repository-name>` with the name of the folder that was created by the cloning process.

4. **Check Available Branches** (optional): 
   ```
   git branch -a
   ```
   This command lists all branches in your repository.

5. **Switch to the 'article-labeling' Branch**: 
   ```
   git checkout article-labeling
   ```
   If the branch doesn’t exist locally, Git will create it.

### Making Changes and Committing

1. **Make Your Changes**: Use any text editor or IDE to modify or add files in the repository folder.

2. **Stage Changes for Commit**: After making changes, stage them with:
   ```
   git add .
   ```
   This command stages all changed files. For specific files, replace `.` with the file name.

3. **Commit the Changes**: 
   ```
   git commit -m "Your commit message"
   ```
   Replace `"Your commit message"` with a meaningful description of the changes.

### Pushing Changes to GitHub

1. **Push to Remote Repository**: 
   ```
   git push origin article-labeling
   ```
   This command pushes your commits to the `article-labeling` branch on the remote repository.

2. **Verify the Push** (optional): 
   ```
   git log --oneline
   ```
   This shows the commit history. Check that your latest commit appears at the top.

### Additional Tips

- **Pull Latest Changes** (if needed): Before starting work, it's a good practice to pull the latest changes from the remote repository:
  ```
  git pull origin article-labeling
  ```

- **Resolving Merge Conflicts**: If there are conflicts after pulling, you'll need to resolve them manually in the conflicting files and then commit the changes. This can happen if someone else changed the same file of the remote repository in the meantime.

- **Regular Commits**: It's a good practice to commit often with descriptive messages. This helps track changes and understand the history of the project.

