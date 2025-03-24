# Dictionary with the question and the respective query, result and expected answer
queries = {
    "question_2": {
        "question": "How many unique product IDs are there in the table?", 
        "write_query": {"query": "SELECT COUNT(DISTINCT product_id) FROM amazon_cleaned"}, 
        "execute_query": {"result": "(1351,)"}, 
        "generate_answer": {"answer": "There are 1351 records in the database."}
    },
    "question_3": {
        "question": "What is the rating of the first product in the table?", 
        "write_query": {"query": "SELECT rating FROM amazon_cleaned LIMIT 1"}, 
        "execute_query": {"result": "[(3.8,)]"}, 
        "generate_answer": {"answer": "The rating of the first product in the table is 3.8."}
    },
    "question_4": {
        "question": "How many unique values for 'category' are in the database?", 
        "write_query": {"query": "SELECT COUNT(DISTINCT category) FROM amazon_cleaned"}, 
        "execute_query": {"result": "[(9,)]"}, 
        "generate_answer": {"answer": "There are 9 unique values for 'category' in the database."}
    },
    "question_5": {
        "question": "What is the maximum value in the column 'discount_percentage' in the table?", 
        "write_query": {"query": "SELECT MAX(discount_percentage) FROM amazon_cleaned"}, 
        "execute_query": {"result": "[('94%',)]"}, 
        "generate_answer": {"answer": "The maximum value in the column 'discount_percentage' in the table is 94%."}
    },
}