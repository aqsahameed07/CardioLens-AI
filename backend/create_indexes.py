from api.database import users_collection

users_collection.create_index(
    "email",
    unique=True,
)

print("MongoDB indexes created successfully.")