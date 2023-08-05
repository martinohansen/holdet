def jaccard_similarity(str1: str, str2: str) -> float:
    # Convert the strings to sets of characters
    set1 = set(str1)
    set2 = set(str2)

    # Calculate the size of the intersection of the sets
    intersection = set1 & set2
    intersection_size = len(intersection)

    # Calculate the size of the union of the sets
    union = set1 | set2
    union_size = len(union)

    # Calculate the Jaccard similarity
    similarity = intersection_size / union_size

    return similarity
