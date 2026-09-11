from src.router.query_router import QueryRoute, QueryRouter


def main():
    print("=" * 60)
    print("QUERY ROUTER TEST")
    print("=" * 60)

    router = QueryRouter()

    test_cases = [
        # ==================================================
        # DOCUMENT ROUTES
        # ==================================================

        (
            "Who manages Project Phoenix?",
            QueryRoute.DOCUMENT,
        ),

        (
            "What is the leave policy?",
            QueryRoute.DOCUMENT,
        ),

        (
            "According to employee policy, what is the rule?",
            QueryRoute.DOCUMENT,
        ),

        # ==================================================
        # GENERAL ROUTES
        # ==================================================

        (
            "What is the capital of France?",
            QueryRoute.GENERAL,
        ),

        (
            "Explain machine learning.",
            QueryRoute.GENERAL,
        ),

        # ==================================================
        # CURRENT / WEB ROUTES
        # ==================================================

        (
            "What is the current Bitcoin price?",
            QueryRoute.CURRENT,
        ),

        (
            "What is the latest AI news?",
            QueryRoute.CURRENT,
        ),

        (
            "What is the weather today?",
            QueryRoute.CURRENT,
        ),

        # ==================================================
        # BOTH ROUTES
        # ==================================================

        (
            "Based on Project Phoenix, how does it compare "
            "with the latest AI assistant trends?",
            QueryRoute.BOTH,
        ),

        (
            "According to our policy, what is the latest "
            "change?",
            QueryRoute.BOTH,
        ),
    ]

    passed = 0

    print()

    for query, expected_route in test_cases:

        actual_route = router.classify(query)

        status = "✅ PASSED" if actual_route == expected_route else "❌ FAILED"

        print(f"Query    : {query}")
        print(f"Expected : {expected_route.value}")
        print(f"Actual   : {actual_route.value}")
        print(f"Result   : {status}")
        print("-" * 60)

        if actual_route == expected_route:
            passed += 1

    print()
    print("=" * 60)
    print(
        f"ROUTER TEST RESULT: "
        f"{passed}/{len(test_cases)} PASSED"
    )
    print("=" * 60)

    assert passed == len(test_cases), (
        f"{len(test_cases) - passed} router test(s) failed."
    )

    print("\n✅ ALL QUERY ROUTER TESTS PASSED")


if __name__ == "__main__":
    main()