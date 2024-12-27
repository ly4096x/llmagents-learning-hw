from autogen import ConversableAgent, GroupChat, GroupChatManager
import sys
import os
import math
import functools
import re


def get_reviews_dict() -> dict[str, list[str]]:
    """
    This function reads in the restaurant data and returns a dictionary that maps each restaurant name to its reviews.
    The output is a dictionary where the key is the restaurant name and the value is a list of reviews for that restaurant.
    Example:
        get_reviews_dict() -> {"applebee s": ["The food at Applebee's was average...",...], "mcdonald s": [...]}
    """
    data_dict = {}

    try:
        with open("restaurant-data.txt", "r") as f:
            for line in f:
                restaurant, review = line.strip().split(".", 1)
                restaurant = restaurant.lower()
                restaurant = re.sub("[^0-9a-zA-Z]", " ", restaurant)
                if restaurant not in data_dict:
                    data_dict[restaurant] = [review]
                else:
                    data_dict[restaurant].append(review)
    except FileNotFoundError as e:
        print("File not found")
        sys.exit()
    return data_dict


def fetch_restaurant_data(restaurant_name: str) -> dict[str, list[str]]:
    # TODO
    # This function takes in a restaurant name and returns the reviews for that restaurant.
    # The output should be a dictionary with the key being the restaurant name and the value being a list of reviews for that restaurant.
    # The "data fetch agent" should have access to this function signature, and it should be able to suggest this as a function call.
    # Example:
    # > fetch_restaurant_data("applebee s")
    # {"applebee s": ["The food at Applebee's was average, with nothing particularly standing out.", ...]}

    return get_reviews_dict()[restaurant_name]


def calculate_overall_score(
    restaurant_name: str, food_scores: list[int], customer_service_scores: list[int]
) -> dict[str, float]:
    # TODO
    # This function takes in a restaurant name, a list of food scores from 1-5, and a list of customer service scores from 1-5
    # The output should be a score between 0 and 10, which is computed as the following:
    # SUM(sqrt(food_scores[i]**2 * customer_service_scores[i]) * 1/(N * sqrt(125)) * 10
    # The above formula is a geometric mean of the scores, which penalizes food quality more than customer service.
    # Example:
    # > calculate_overall_score("applebee s", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    # {"applebee s": 5.048}
    # NOTE: be sure to that the score includes AT LEAST 3  decimal places. The public tests will only read scores that have
    # at least 3 decimal places.

    # Ensure both lists have the same length
    if len(food_scores) != len(customer_service_scores):
        raise ValueError(
            "food_scores and customer_service_scores must be of the same length."
        )

    N = len(food_scores)
    if N == 0:
        return {restaurant_name: 0.000}

    sum_val = 0.0
    for i in range(N):
        sum_val += math.sqrt((food_scores[i] ** 2) * customer_service_scores[i])

    # Scale factor
    scale = (1 / (N * math.sqrt(125))) * 10
    raw_score = sum_val * scale
    # Round to at least 3 decimal places:
    score = round(raw_score, 3)

    return {restaurant_name: score}


def llm_config(*, use_gemini=False):
    if use_gemini:
        return {
            "config_list": [
                {
                    "model": "gemini-1.5-flash",
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                },
            ],
        }
    else:
        return {
            "config_list": [
                {
                    "model": "gpt-4o-mini",
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                },
            ],
        }


# Do not modify the signature of the "main" function.
def main(user_query: str):
    # Example system message for the entrypoint agent
    entrypoint_agent_system_message = (
        "You are the primary agent. The user wants information about a restaurant. "
        "Coordinate with other agents if needed to fetch data or calculate scores. When you are ready to answer the user's query, note that the returned float must have at least 3 decimal places. "
        "When you feel the query has been answered, you will say '[END CHAT]' to terminate the discussion."
    )

    # the main entrypoint/supervisor agent
    entrypoint_agent = ConversableAgent(
        "entrypoint_agent",
        system_message=entrypoint_agent_system_message,
        llm_config=llm_config(),
    )

    # TODO
    # Create more agents here.

    restaurant_name_extractor_agent = ConversableAgent(
        "restaurant_name_extractor_agent",
        llm_config=llm_config(),
        system_message="You are a extractor agent that extracts the restaurant name from the input. You always output the lower case restaurant name, and replacing each non alpha-numeric special character in the name with a whitespace. The output format is: 'Restaurant name: <RestaurantName>.'. If you can't do the job, say that you can't do it and explain why.",
    )

    # 2) (Optional) Create a dedicated data fetch agent
    data_fetch_agent_system_message = "You are a data fetch agent. You can fetch restaurant data using the 'fetch_restaurant_data' function. You always call the function with a lower case restaurant name, and replacing each non alpha-numeric special character in the name with a whitespace. If there are any problems, say that you can't do it and explain why."
    data_fetch_agent = ConversableAgent(
        "data_fetch_agent",
        system_message=data_fetch_agent_system_message,
        llm_config=llm_config(),
    )
    data_fetch_agent.register_for_llm(
        name="fetch_restaurant_data",
        description="Fetches the reviews for a specific restaurant.",
    )(fetch_restaurant_data)

    scoring_agent = ConversableAgent(
        "scoring_agent",
        llm_config=llm_config(),
        system_message="""You are a scoring agent. You should look at every single review for a restaurant and extract two scores:
- `food_score`: the quality of food at the restaurant. This will be a score between 1 and 5.
- `customer_service_score`: the quality of customer service at the restaurant. This will be a score between 1 and 5.

You should extract these two scores by looking for keywords in the review. Each review has keyword adjectives that correspond to the score that the restaurant should get for its `food_score` and `customer_service_score`. Here are the keywords you should look out for:

- Score 1/5 if the review has one of these adjectives: awful, horrible, or disgusting.
- Score 2/5 if the review has one of these adjectives: bad, unpleasant, or offensive.
- Score 3/5 if the review has one of these adjectives: average, uninspiring, or forgettable.
- Score 4/5 if the review has one of these adjectives: good, enjoyable, or satisfying.
- Score 5/5 if the review has one of these adjectives: awesome, incredible, or amazing.

You return two arrays: `food_scores` and `customer_service_scores`. The length of each array should be the same as the number of reviews in the query. Each element of these arrays should correspond to a review.
""",
    )

    # 3) (Optional) Create a scoring agent
    overall_scoring_agent_system_message = "You are a scoring agent. You calculate an overall score based on food and customer service scores using the `calculate_overall_score` function."
    overall_scoring_agent = ConversableAgent(
        "overall_scoring_agent",
        system_message=overall_scoring_agent_system_message,
        llm_config=llm_config(),
    )
    overall_scoring_agent.register_for_llm(
        name="calculate_overall_score",
        description="Calculates an overall score for a restaurant.",
    )(calculate_overall_score)

    entrypoint_agent.register_for_execution()(calculate_overall_score)
    entrypoint_agent.register_for_execution()(fetch_restaurant_data)

    # TODO
    # Fill in the argument to `initiate_chats` below, calling the correct agents sequentially.
    # If you decide to use another conversation pattern, feel free to disregard this code.

    # Uncomment once you initiate the chat with at least one agent.
    if 0:  # use linear conversation

        conversation = [
            {
                "sender": entrypoint_agent,
                "recipient": restaurant_name_extractor_agent,
                "max_turns": 2,
                "message": f"The user has asked '{user_query}'. Give me the name of the restaurant. The name will be used later in the chat.",
            },
            {
                "sender": entrypoint_agent,
                "recipient": data_fetch_agent,
                "max_turns": 2,
                "message": (
                    f"Let's fetch the reviews for the restaurant using the data fetch agent."
                ),
            },
            {
                "sender": entrypoint_agent,
                "recipient": scoring_agent,
                "max_turns": 2,
                "message": "Give me the arrays of food and customer service scores from the reviews.",
            },
            {
                "sender": entrypoint_agent,
                "recipient": overall_scoring_agent,
                "max_turns": 2,
                "message": (
                    f"Calculate the overall score for the restaurant that the user asked for."
                ),
            },
        ]

        result = entrypoint_agent.initiate_chats(conversation)
    else:  # use group chat
        groupchat = GroupChat(
            agents=[
                entrypoint_agent,
                overall_scoring_agent,
                data_fetch_agent,
                scoring_agent,
                restaurant_name_extractor_agent,
            ],
            messages=[],
            max_round=20,
            speaker_selection_method="auto",
        )

        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=llm_config(),
            is_termination_msg=lambda msg: "[END CHAT]" in msg["content"],
        )
        result = entrypoint_agent.initiate_chat(
            manager,
            message=f"User is asking '{user_query}'. Give me the overall score for the restaurant that the user asked for. We need to extract the restaurant name first, then use the name to look up reviews for that restaurant, and then get the food and customer service scores for each of the reviews, and last we calculate the overall score.",
        )

    # print("Result score: ", result[-1]["content"])

    # Print the entire conversation result
    print()
    print()
    print(result)


# DO NOT modify this code below.
if __name__ == "__main__":
    assert (
        len(sys.argv) > 1
    ), "Please ensure you include a query for some restaurant when executing main."
    main(sys.argv[1])
