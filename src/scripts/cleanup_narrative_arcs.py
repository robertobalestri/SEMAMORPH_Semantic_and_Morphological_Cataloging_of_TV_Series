import os
import shutil

def delete_narrative_arcs(series: str, season: str):
    base_dir = "data"  # Base directory for data
    chroma_db_path = os.path.join('chroma_db')  # Path to the @chroma_db folder
    database_path = os.path.join('narrative_arcs.db')

    # Delete the database file if it exists
    if os.path.exists(database_path):
        os.remove(database_path)
        print(f"Deleted file: {database_path}")
    else:
        print(f"File not found: {database_path}")

    # Delete the @chroma_db folder if it exists
    if os.path.exists(chroma_db_path):
        shutil.rmtree(chroma_db_path)
        print(f"Deleted folder: {chroma_db_path}")
    else:
        print(f"Folder not found: {chroma_db_path}")

    # Construct the path for the season directory
    season_path = os.path.join(base_dir, series, season)

    # Check if the season directory exists
    if os.path.exists(season_path):
        # Delete the multiagent season narrative arcs JSON file
        season_narrative_arcs_path = os.path.join(season_path, f"{series}{season}_multiagent_season_narrative_arcs.json")
        if os.path.exists(season_narrative_arcs_path):
            os.remove(season_narrative_arcs_path)
            print(f"Deleted file: {season_narrative_arcs_path}")
        else:
            print(f"File not found: {season_narrative_arcs_path}")

        # Delete the multiagent season narrative analysis file
        season_narrative_analysis_path = os.path.join(season_path, f"{series}{season}_multiagent_season_narrative_analysis.txt")
        if os.path.exists(season_narrative_analysis_path):
            #os.remove(season_narrative_analysis_path)
            print(f"Deleted file: {season_narrative_analysis_path}")
        else:
            print(f"File not found: {season_narrative_analysis_path}")

        # Delete the multiagent episode narrative arcs JSON files
        for episode in os.listdir(season_path):
            episode_path = os.path.join(season_path, episode)
            if os.path.isdir(episode_path):
                episode_narrative_arcs_path = os.path.join(episode_path, f"{series}{season}{episode}_multiagent_episode_narrative_arcs.json")
                if os.path.exists(episode_narrative_arcs_path):
                    os.remove(episode_narrative_arcs_path)
                    print(f"Deleted file: {episode_narrative_arcs_path}")
                else:
                    print(f"File not found: {episode_narrative_arcs_path}")

        # Delete the multiagent episode narrative analysis files
        for episode in os.listdir(season_path):
            episode_path = os.path.join(season_path, episode)
            if os.path.isdir(episode_path):
                episode_narrative_analysis_path = os.path.join(episode_path, f"{series}{season}{episode}_multiagent_episode_narrative_analysis.txt")
                if os.path.exists(episode_narrative_analysis_path):
                    #os.remove(episode_narrative_analysis_path)
                    print(f"Deleted file: {episode_narrative_analysis_path}")
                else:
                    print(f"File not found: {episode_narrative_analysis_path}")
    else:
        print(f"Season directory not found: {season_path}")

if __name__ == "__main__":
    # Replace these with the actual series and season names you want to clean up
    series_name = "GA"  # e.g., "MyShow"
    season_name = "S01"  # e.g., "Season1"
    
    delete_narrative_arcs(series_name, season_name)
