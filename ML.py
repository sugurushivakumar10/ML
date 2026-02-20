

for col in X_open.select_dtypes(include=["object"]).columns:
    if col in label_encoders:

        le = label_encoders[col]

        # Replace unseen labels with "Unknown"
        X_open[col] = X_open[col].apply(
            lambda x: x if x in le.classes_ else "Unknown"
        )

        # If "Unknown" not in training classes, add it
        if "Unknown" not in le.classes_:
            le.classes_ = np.append(le.classes_, "Unknown")

        X_open[col] = le.transform(X_open[col])
