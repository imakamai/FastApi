def individual_data(user):
    return  {
        "id": str(user["_id"]),
        "username": user["username"],
        "password": user["password"],
        "accountId": user["accountId"],
        "apiKey": user["api_key"],
    }