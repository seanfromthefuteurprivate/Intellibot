import logging

logger = logging.getLogger(__name__)

def score_tickers(mentions, market_data):
    """
    Scores and ranks tickers based on mentions and market momentum.
    Returns a list of (ticker, score) tuples, sorted by score desc.
    """
    logger.info("Scoring tickers...")
    scores = []
    
    for ticker in mentions:
        # Placeholder scoring logic
        score = 0
        if ticker in market_data:
            score += 10 # Base score
            score += market_data[ticker]["change"] * 100
            
        scores.append((ticker, score))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
