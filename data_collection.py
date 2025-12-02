import steamreviews
import pandas as pd
import json
import time
from pathlib import Path

class SteamReviewCollector:
    def __init__(self, output_dir='data/raw'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_reviews_for_game(self, app_id, game_name, max_reviews=1000):
        """Collect reviews for a single game"""
        print(f"Collecting reviews for {game_name} (App ID: {app_id})...")
        
        request_params = dict()
        request_params['filter'] = 'all'  # Get all reviews
        request_params['language'] = 'english'
        request_params['day_range'] = '365'  # Last year
        
        reviews_dict, query_count = steamreviews.download_reviews_for_app_id(
            app_id,
            chosen_request_params=request_params
        )
        
        # Convert to list of review dicts
        reviews = []
        for review_id, review_data in reviews_dict['reviews'].items():
            reviews.append({
                'game_name': game_name,
                'app_id': app_id,
                'review_id': review_id,
                'review_text': review_data['review'],
                'voted_up': review_data['voted_up'],
                'votes_up': review_data['votes_up'],
                'votes_funny': review_data['votes_funny'],
                'weighted_vote_score': review_data['weighted_vote_score'],
                'playtime_forever': review_data['author']['playtime_forever'],
                'playtime_at_review': review_data['author']['playtime_at_review'],
                'timestamp_created': review_data['timestamp_created'],
                'timestamp_updated': review_data['timestamp_updated'],
            })
            
            if len(reviews) >= max_reviews:
                break
        
        # Save to file
        output_file = self.output_dir / f"{app_id}_{game_name.replace(' ', '_')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, indent=2, ensure_ascii=False)
        
        print(f"Collected {len(reviews)} reviews, saved to {output_file}")
        return reviews
    
    def collect_multiple_games(self, game_list, reviews_per_game=1000):
        """
        Collect reviews for multiple games
        game_list: List of tuples (app_id, game_name)
        """
        all_reviews = []
        
        for app_id, game_name in game_list:
            try:
                reviews = self.collect_reviews_for_game(app_id, game_name, reviews_per_game)
                all_reviews.extend(reviews)
                time.sleep(2)  # Be nice to Steam's API
            except Exception as e:
                print(f"Error collecting reviews for {game_name}: {e}")
                continue
        
        # Save combined dataset
        df = pd.DataFrame(all_reviews)
        df.to_csv(self.output_dir / 'all_reviews.csv', index=False)
        print(f"\nTotal reviews collected: {len(all_reviews)}")
        return df

# Popular games list (App ID, Name)
POPULAR_GAMES = [
    (730, "Counter-Strike 2"),
    (570, "Dota 2"),
    (578080, "PUBG: BATTLEGROUNDS"),
    (2807960, "Battlefield\u2122 6"),
    (431960, "Wallpaper Engine"),
    (1808500, "ARC Raiders"),
    (1172470, "Apex Legends\u2122"),
    (3241660, "R.E.P.O."),
    (3167020, "Escape From Duckov"),
    (236390, "War Thunder"),
    (2767030, "Marvel Rivals"),
    (3527290, "PEAK"),
    (1203220, "NARAKA: BLADEPOINT"),
    (2507950, "Delta Force"),
    (381210, "Dead by Daylight"),
    (322170, "Geometry Dash"),
    (413150, "Stardew Valley"),
    (252490, "Rust"),
    (230410, "Warframe"),
    (271590, "Grand Theft Auto V Legacy"),
    (105600, "Terraria"),
    (3240220, "Grand Theft Auto V Enhanced"),
    (359550, "Tom Clancy's Rainbow Six\u00ae Siege X"),
    (1973530, "Limbus Company"),
    (1086940, "Baldur's Gate 3"),
    (227300, "Euro Truck Simulator 2"),
    (739630, "Phasmophobia"),
    (1938090, "Call of Duty\u00ae"),
    (1174180, "Red Dead Redemption 2"),
    (238960, "Path of Exile"),
    (2357570, "Overwatch\u00ae 2"),
    (3419430, "Bongo Cat"),
    (550, "Left 4 Dead 2"),
    (438100, "VRChat"),
    (322330, "Don't Starve Together"),
    (3405690, "EA SPORTS FC\u2122 26"),
    (553850, "HELLDIVERS\u2122 2"),
    (3405340, "Megabonk"),
    (1030300, "Hollow Knight: Silksong"),
    (1449850, "Yu-Gi-Oh! Master Duel"),
    (3551340, "Football Manager 26"),
    (284160, "BeamNG.drive"),
    (394360, "Hearts of Iron IV"),
    (1091500, "Cyberpunk 2077"),
    (1281930, "tModLoader"),
    (3224770, "Umamusume: Pretty Derby"),
    (250900, "The Binding of Isaac: Rebirth"),
    (3450310, "Europa Universalis V"),
    (252950, "Rocket League\u00ae"),
    (440, "Team Fortress 2"),
    (1222670, "The Sims\u2122 4"),
    (1366800, "Crosshair X"),
    (367520, "Hollow Knight"),
    (3949040, "RV There Yet?"),
    (289070, "Sid Meier\u2019s Civilization\u00ae VI"),
    (2592160, "Dispatch"),
    (1364780, "Street Fighter\u2122 6"),
    (1245620, "ELDEN RING"),
    (2073620, "Arena Breakout: Infinite"),
    (1623730, "Palworld"),
    (1158310, "Crusader Kings III"),
    (489830, "The Elder Scrolls V: Skyrim Special Edition"),
    (2300320, "Farming Simulator 25"),
    (1422450, "Deadlock"),
    (2669320, "EA SPORTS FC\u2122 25"),
    (4000, "Garry's Mod"),
    (291550, "Brawlhalla"),
    (2246340, "Monster Hunter Wilds"),
    (108600, "Project Zomboid"),
    (2584990, "Shadowverse: Worlds Beyond"),
    (1665460, "eFootball\u2122"),
    (3354750, "skate."),
    (1203620, "Enshrouded"),
    (221100, "DayZ"),
    (594650, "Hunt: Showdown 1896"),
    (244210, "Assetto Corsa"),
    (377160, "Fallout 4"),
    (646570, "Slay the Spire"),
    (1771300, "Kingdom Come: Deliverance II"),
    (960090, "Bloons TD 6"),
    (1905180, "OBS Studio"),
    (1145350, "Hades II"),
    (2379780, "Balatro"),
    (3513350, "Wuthering Waves"),
    (892970, "Valheim"),
    (275850, "No Man's Sky"),
    (1142710, "Total War: WARHAMMER III"),
    (714010, "Aimlabs"),
    (251570, "7 Days to Die"),
    (2642680, "Half Sword Tech Demo"),
    (1551360, "Forza Horizon 5"),
    (261550, "Mount & Blade II: Bannerlord"),
    (813780, "Age of Empires II: Definitive Edition"),
    (2139460, "Once Human"),
    (629520, "Soundpad"),
    (1326470, "Sons Of The Forest"),
    (2073850, "THE FINALS"),
    (294100, "RimWorld"),
    (2622380, "ELDEN RING NIGHTREIGN"),
    (1085660, "Destiny 2"),
]

if __name__ == "__main__":
    collector = SteamReviewCollector()
    df = collector.collect_multiple_games(POPULAR_GAMES, reviews_per_game=1000)
    print(f"Dataset shape: {df.shape}")
    print(df.head())