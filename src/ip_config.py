from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IPConfig:
    canonical_name: str
    aliases: list[str]
    reddit_subreddits: list[str]
    youtube_queries: list[str]
    bluesky_queries: list[str]


PRIMARY_IPS: list[IPConfig] = [
    IPConfig(
        canonical_name="Star Wars",
        aliases=["star wars", "the mandalorian", "jedi", "sith"],
        reddit_subreddits=["StarWars", "TheMandalorianTV", "StarWarsCantina"],
        youtube_queries=["Star Wars", "The Mandalorian", "Andor", "Ahsoka", "new star wars"],
        bluesky_queries=["Star Wars", "The Mandalorian", "Andor", "Ahsoka", "new star wars"],
    ),
    IPConfig(
        canonical_name="Pokemon",
        aliases=["pokemon", "pokémon", "pikachu", "pokemon tcg"],
        reddit_subreddits=["pokemon", "TheSilphRoad", "PokemonTCG"],
        youtube_queries=["Pokemon", "Pokemon TCG",  "Pokemon Scarlet Violet"],
        bluesky_queries=["Pokemon", "Pokemon TCG", "Pokemon Scarlet Violet"],
    ),
    IPConfig(
        canonical_name="Dungeons & Dragons",
        aliases=["dungeons and dragons", "d&d", "dnd", "baldur's gate 3", "critical role", "dimension 20"],
        reddit_subreddits=["DnD", "dndnext", "BaldursGate3", "CriticalRole", "Dimension20"],
        youtube_queries=["Dungeons and Dragons", "DND actual play", "Baldur's Gate 3", "Critical Role", ],
        bluesky_queries=["Dungeons and Dragons", "DND actual play", "Baldur's Gate 3", "Critical Role", ],
    ),
    IPConfig(
        canonical_name="Harry Potter",
        aliases=["harry potter", "hogwarts", "wizarding world"],
        reddit_subreddits=["harrypotter", "HPfanfiction", "HogwartsLegacyGaming"],
        youtube_queries=["Harry Potter", "Wizarding World", "Hogwarts Legacy"],
        bluesky_queries=["Harry Potter", "Wizarding World", "Hogwarts Legacy"],
    ),
    IPConfig(
        canonical_name="Marvel",
        aliases=["marvel", "mcu", "avengers", "spider-man"],
        reddit_subreddits=["marvelstudios", "Marvel", "Spiderman"],
        youtube_queries=["Marvel", "MCU", "Avengers", "Spider-Man"],
        bluesky_queries=["Marvel", "MCU", "Avengers", "Spider-Man"],
    ),
    IPConfig(
        canonical_name="Warhammer 40K",
        aliases=["warhammer 40k", "40k", "warhammer", "space marines"],
        reddit_subreddits=["Warhammer40k", "Grimdank", "40kLore"],
        youtube_queries=["Warhammer 40K", "40k lore", "Space Marines", "Warhammer theory", "Adeptus Astartes"],
        bluesky_queries=["Warhammer 40K", "40k lore", "Space Marines", "Warhammer theory", "Adeptus Astartes"],
    ),
]


def build_alias_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for ip_config in PRIMARY_IPS:
        for alias in ip_config.aliases:
            rows.append(
                {
                    "ip_name": ip_config.canonical_name,
                    "alias": alias,
                }
            )
    return rows
