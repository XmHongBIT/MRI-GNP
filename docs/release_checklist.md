# Release Checklist

- Remove any private dataset paths, hospital names, and internal usernames.
- Confirm that no patient-identifiable data is stored in `.npz`, `.csv`, `.json`, or checkpoint files.
- Add a real open-source license before publishing.
- Add a short model card if checkpoints will be uploaded.
- Replace placeholder configs with one verified public training recipe.
- Check `.gitignore` before the first push so cached samples and output folders are not uploaded.
