"""Direction Controls — Fix: genre, situation, motion, look, dialogue and design choices; heat, pace, cuts and violence come from the request.

The node only writes plain `Label: value` lines. The director Fix node
(story_director_fix.py) reads them and carries their meaning into its rules.
Everything the original node emits is unchanged; the new lines are additions.
"""
from __future__ import annotations

from comfy_api.latest import io

from .story_director import DIALOGUE_OPTIONS, GENRES, MOTION_STYLES, VISUAL_LOOKS

# Looks / motions that already imply a single fixed camera: no cuts unless the user asks for them.
SINGLE_CAMERA_LOOKS = frozenset({
    "CCTV / Security Camera", "Bodycam", "Webcam / Livestream", "Laptop / PC Webcam",
    "Video Call / Conference Webcam", "Dashcam / Vehicle Camera", "Doorbell / Fixed Home Camera",
    "Surveillance / Hidden Camera", "Found Footage",
    "Screenlife / Desktop Capture", "Creator Bedroom Camera", "Casting / Audition Tape",
})
SINGLE_CAMERA_MOTIONS = frozenset({"Static Tableau"})  # Locked-Off only fixes the camera inside each shot

HIGH_ACTION_MOTIONS = frozenset({
    "Fast Motion", "Intense Dynamic Motion", "Hyperlapse",
    "Chaotic Erratic Motion", "Frenetic Kinetic Motion", "Parkour Pursuit",
    "Staccato Music-Video Motion"})  # camera-only moves (whip pan, crash zoom) never set body activity
LOW_ACTION_MOTIONS = frozenset({
    "Slow Motion", "Super Slow Motion", "Minimal Motion", "Meditative Calm Motion",
    "Gentle Organic Motion", "Dreamlike Floating Motion", "Intimate Micro-Motion"})
# camera-only styles never decide how much the bodies do
ACTION_CHOICES = ["Auto", "Short", "Normal", "Long", "Very Long"]
VIDEO_TASKS = ["Swap", "Edit", "Add", "Remove"]
ERA_CHOICES = ["Auto", "Present day", "Stone Age", "Ancient Egypt", "Classical Antiquity (Greece / Rome)",
               "Medieval", "Feudal Japan", "Pre-Columbian Americas", "Renaissance",
               "18th Century (Baroque / Rococo)", "Victorian", "Belle Époque / Edwardian", "1920s", "1930s", "1940s", "1950s", "1960s", "1970s", "1980s",
               "1990s", "2000s", "2010s"]
VERY_HIGH_ACTION_MOTIONS = frozenset({"Super Fast Motion"})

PROGRESSION_CHOICES = ["Auto", "Slow", "Normal", "Gradual", "Fast", "Very Fast", "Instant", "Reverse"]
INTENSITY_LEVELS = ["Auto", "Flirt", "Sensual", "Heated", "Sex", "Hardcore", "Extreme"]
NUDITY_LEVELS = ["Auto", "Clothed", "Underwear", "Topless / Partial", "Nude", "Explicit"]
VIOLENCE_LEVELS = ["Auto", "Mild", "Moderate", "Strong", "Graphic", "Extreme"]
SINGLE_CAMERA_GENRES = frozenset({"Webcam / Cam Show"})
ADULT_NUDITY = frozenset({"Underwear", "Topless / Partial", "Nude", "Explicit"})
# adult genres and looks are gone: any genre becomes its adult version when the request is adult
NEW_GENRES = ["POV / First-Person", "Cosplay / Roleplay", "Webcam / Cam Show", "Boudoir / Glamour Shoot",
              "Home Video / Amateur"]
GENRE_CHOICES = [g for g in GENRES if "18+" not in str(g) and str(g).lower() not in ("auto", "none")] + [g for g in NEW_GENRES if g not in GENRES]
MOTION_CHOICES = [m for m in MOTION_STYLES if str(m).lower() not in ("auto", "none")]
LOOK_CHOICES = [k for k in VISUAL_LOOKS if "18+" not in str(k) and str(k).lower() not in ("auto", "none")]
COVERAGE_CHOICES = ["Auto", "Single take", "Calm (10-15s cuts)", "Moderate (6-8s cuts)",
                    "Dynamic (3-5s cuts)"]
OUTFIT_STYLES = [
    "Casual", "Urban Streetwear", "Y2K", "Ultra Low-Rise", "Hip-Hop", "Skater", "Grunge", "Punk / Rock", "Goth",
    "Alt / E-Girl", "Boho", "Minimalist",
    "Sophisticated", "Old Money / Quiet Luxury", "Cocktail Dress", "Evening Gown", "Red Carpet Glam",
    "Business / Office", "Power Suit",
    "Yoga", "Gym / Sportswear", "Athleisure", "Running", "Tennis", "Swimwear / Bikini", "Cheerleader (adult)",
    "Maid", "Nurse", "Police", "Flight Attendant", "Military", "Secretary", "Waitress / Bartender",
    "Idol (K-pop stage)", "Cosplay / Anime", "Cyberpunk", "Steampunk", "Fantasy / Elf", "Superhero", "Pirate",
    "Lingerie", "Corset / Bustier", "Latex / Leather", "Clubwear / Bodycon", "Fishnet / Rave", "Silk Robe / Kimono",
    "Babydoll",
    "Western / Cowgirl", "Festival", "Tropical / Beachwear"]
SITUATIONS = [
    'First Date',
    'Blind Date',
    'Meet-Cute',
    'Anniversary Dinner',
    'Proposal',
    'Wedding Night',
    'Ex Reunion',
    'Breakup',
    'Make-Up After a Fight',
    'Jealousy',
    'Caught Cheating',
    'Dentist',
    'Doctor Check-Up',
    'Massage / Spa',
    'Hair Salon',
    'Tattoo Studio',
    'Personal Trainer',
    'Job Interview',
    'Casting / Audition',
    'Photo Shoot',
    'Tailor Fitting',
    'House Viewing',
    'Delivery at the Door',
    'Plumber / Repair Visit',
    'Room Service',
    'New Neighbor',
    'Landlord Visit',
    'Pool Cleaner',
    'Robbery / Hold-Up',
    'Heist',
    'Interrogation',
    'Traffic Stop',
    'Hostage Standoff',
    'Chase',
    'Blackmail',
    'Stakeout',
    'Undercover',
    'House Party',
    'Nightclub',
    'Bar Pickup',
    'Office Party',
    'Karaoke Night',
    'Game Night',
    'Bachelor / Bachelorette Party',
    'Strip Club',
    'Morning After',
    'Cooking Together',
    'Moving Day',
    'Power Outage',
    'Stuck in an Elevator',
    'Locked Out',
    'Rainy Day In',
    'Road Trip',
    'Hitchhiker',
    'Train Compartment',
    'Flight',
    'Hotel Check-In',
    'Beach Day',
    'Camping',
    'Gas Station',
    'Late at the Office',
    'Boss and Employee',
    'Live Stream',
    'Podcast Interview',
    'Backstage',
    'Music Video Shoot',
    'Gym Session',
    'Yoga Class',
    'Boxing Training',
    'Locker Room',
    'Haunted House',
    "Vampire's Visit",
    "Witch's Spell",
    'Alien Encounter',
    'Dating App Hookup',
    'One-Night Stand',
    'Booty Call',
    'Hotel Affair',
    'Sugar Date',
    'Escort Booking',
    'Seductive Neighbor',
    'Delivery Seduction',
    'Strip Tease',
    'Lap Dance',
    'Private Dance',
    'Webcam Show',
    'Content Creator Shoot',
    'Boudoir Photo Shoot',
    'Casting Couch',
    'Pole Dance Class',
    'Strip Poker',
    'Truth or Dare',
    'Roleplay Night',
    'Body Painting',
    'Sex Toy Unboxing',
    'Shower Together',
    'Hot Tub / Sauna',
    'Skinny Dipping',
    'Car / Back Seat',
    'Elevator Quickie',
    'Mile High Club',
    'Fitting Room',
    'Sex Shop Visit',
    'Morning Sex',
    'Honeymoon',
    'Makeup Sex',
    'Movie Night on the Couch',
    'Massage with Happy Ending',
    'Sunbathing / Tan Lines',
    'BDSM Dungeon Session',
    'Domination / Submission',
    'Bondage',
    'Cuckold Night',
    'Swingers Party',
    'Exhibitionist Couple',
    'Threesome Invitation',
    'Orgy / Sex Party',
    'Bachelor Party Show',
]
ADULT_SITUATIONS = frozenset({'Dating App Hookup', 'One-Night Stand', 'Booty Call', 'Hotel Affair', 'Sugar Date', 'Escort Booking', 'Seductive Neighbor', 'Delivery Seduction', 'Strip Tease', 'Lap Dance', 'Private Dance', 'Webcam Show', 'Content Creator Shoot', 'Boudoir Photo Shoot', 'Casting Couch', 'Pole Dance Class', 'Strip Poker', 'Truth or Dare', 'Roleplay Night', 'Body Painting', 'Sex Toy Unboxing', 'Shower Together', 'Hot Tub / Sauna', 'Skinny Dipping', 'Car / Back Seat', 'Elevator Quickie', 'Mile High Club', 'Fitting Room', 'Sex Shop Visit', 'Morning Sex', 'Honeymoon', 'Makeup Sex', 'Movie Night on the Couch', 'Massage with Happy Ending', 'Sunbathing / Tan Lines', 'BDSM Dungeon Session', 'Domination / Submission', 'Bondage', 'Cuckold Night', 'Swingers Party', 'Exhibitionist Couple', 'Threesome Invitation', 'Orgy / Sex Party', 'Bachelor Party Show'})
DENSITY_CHOICES = ["Auto", "Sparse", "Balanced", "Dense"]
STORY_PACE_CHOICES = ["Auto", "Slow", "Normal", "Fast", "Very Fast"]


def _inactive(value) -> bool:
    return str(value or "").strip().lower() in ("", "none", "auto")


class H3CompactDirectionControlsFix(io.ComfyNode):
    """Direction Controls: genre, situation, look, dialogue and the design choices; everything else comes from the request."""

    @classmethod
    def define_schema(cls):
        def choices(values):
            return ["none", *[value for value in values if str(value).lower() != "none"]]

        return io.Schema(
            node_id="H3CompactDirectionControls",
            display_name="H3 Compact Director — Direction Controls",
            category="text/minimax_h3",
            inputs=[
                io.Combo.Input("genre_1", display_name="Genre", options=["none", "Random", *GENRE_CHOICES], default="none",
                    tooltip="Primary story/world/form/rendering choice. In creative modes it shapes the concrete situation and payoff, not just mood. Faithful edits keep their source story."),
                io.Combo.Input("situation", display_name="Situation", options=["none", "Random", *SITUATIONS], default="none",
                    tooltip="The premise: what is happening and where (a first date, the dentist, a robbery, a strip tease...). Genre sets the tone, Situation the story it plays in: Comedy + Dentist, Horror + First Date. An adult request plays its adult version; the 18+ situations are adult by themselves. Your written request wins on every detail; in Transfer it only dresses the guide's own action. Consenting adults only."),
                io.Combo.Input("motion_style_1", display_name="Motion Style", options=["none", "Random", *MOTION_CHOICES], default="none",
                    tooltip="How the bodies and the camera move (cadence, energy, camera behaviour). none: the director decides from the request. Random: one is drawn each run (your request wins if it says otherwise)."),
                io.Combo.Input("look_1", display_name="Visual Look", options=["none", "Random", *LOOK_CHOICES], default="none",
                    tooltip="Primary capture/colour treatment: camera, film, grade. Looks never change the era, clothes or sets (use Era for that). When a look is selected it also decides the palette."),
                io.Combo.Input("dialogue", options=choices(DIALOGUE_OPTIONS), default="none",
                    tooltip="none leaves dialogue unchanged; No dialogue explicitly requests silence from speakers."),
                io.Combo.Input("dialogue_density", display_name="Dialogue Density",
                    options=DENSITY_CHOICES, default="Auto",
                    tooltip="Sparse = a few short lines. Balanced = each lead speaks once or twice. Dense = a continuous exchange. Auto lets the director decide. Ignored when dialogue is No dialogue."),
                io.Combo.Input("outfit_1", display_name="Outfit Style", options=["none", "Random", *OUTFIT_STYLES], default="none",
                    tooltip="Clothing style. With Wardrobe Auto or Create New the director designs outfits in this style. With Keep Reference the reference clothes are kept (same pieces, colours, fabrics) and restyled to this cut, e.g. the same jeans become low-rise; full uniforms or costumes are made in the reference colours. With Guide Garments + Reference Style it restyles the guide's garments. Keep Guide is not changed. none: no style. Random: one is drawn each run. Your own words about clothes win."),
                io.Combo.Input("wardrobe", display_name="Wardrobe",
                    options=["Auto", "Keep Reference", "Keep Guide", "Create New", "Guide Garments + Reference Style"], default="Auto",
                    tooltip="Auto lets the director choose freely. Keep Reference = the outfit of each character's reference picture. Guide Garments + Reference Style (Transfer) = the same garments as the guide performer (pieces, cut, layers, how they move and come off), restyled in the style of each character's reference (colours, fabrics, patterns, accessories); elsewhere it acts as Keep Reference. Keep Guide = exactly the clothes worn in the guide image/video. Create New = new clothing without changing identity or performance. Clothes, shoes and accessories only (hair and makeup have their own control). Written user instructions win."),
                io.Combo.Input("hair_makeup", display_name="Hair & Makeup",
                    options=["Auto", "Keep Reference", "Keep Guide", "Create New"], default="Auto",
                    tooltip="Hairstyle (cut, length, shape, texture, colour), makeup, nails, facial hair and grooming. Auto lets the director choose (Era, Genre and other controls can restyle it). Keep Reference = as in each character's reference picture. Keep Guide = as on the person in the guide image/video. Create New = newly designed from the request, Era, Genre, NSFW and Visual look. Face, age, tattoos and marks always stay the reference's. Written user instructions win."),
                io.Combo.Input("environment", display_name="Environment",
                    options=["Auto", "Keep Reference", "Keep Guide", "Create New"], default="Auto",
                    tooltip="Auto lets the director choose freely. Keep Reference = the place in the reference picture. Keep Guide = exactly the place shown in the guide image/video. Create New = a new setting without changing cast or actions. Written user instructions win."),
                io.Combo.Input("body_proportions", display_name="Body Proportions",
                    options=["Auto", "Keep Reference", "Keep Guide", "Creative Adaptation"], default="Auto",
                    tooltip="Auto lets the director choose freely. Keep Reference = the body of each reference picture. Keep Guide = exactly the build and proportions of the person in the guide image/video (face stays the reference's). Creative Adaptation = coherent changes to build and proportions without changing facial identity, age, pose or choreography. Written user instructions win."),
                io.Combo.Input("era", display_name="Era",
                    options=["none", "Random", *[e for e in ERA_CHOICES if e != "Auto"]], default="none",
                    tooltip="The period the story's world lives in (historic eras or decades): mainstream fashion of that time as people really wore it, sets, vehicles, technology and objects. No futurism or costume caricature. Auto adds no era rule: the period follows the request, the references and the genre. Independent from the Visual look (a 1990s camera look does not dress anyone in the 90s)."),
                io.Combo.Input("video_task", display_name="Video Task",
                    options=VIDEO_TASKS, default="Swap",
                    tooltip="Transfer and Edit modes. In Edit (Viggle Remake): Swap = the pictures recast people of the video (swap LoRA); Edit = change clothes, objects, background, weather, light or style, people stay themselves (edit LoRA); Add = what the pictures show is added; Remove = what you name disappears (edit LoRA). In Transfer: Swap = the reference pictures replace the people of the guide video (classic Transfer). Transfer Edit / Add / Remove: connect a frame of the guide video as Picture 1; the video stays the base and its people stay themselves. Edit = change only what you ask or the controls set (clothes, hair, colours, objects, light, place); Picture 2-4 can supply a garment, hairstyle, object or style to copy. Add = Picture 2-4 join the video as new people (or objects). Remove = the person or thing you name disappears cleanly."),
            ],
            outputs=[io.String.Output("direction_context")],
        )

    @classmethod
    def execute(cls, genre_1="none", situation="none", motion_style_1="none", look_1="none", dialogue="none", dialogue_density="Auto",
                wardrobe="Auto", hair_makeup="Auto", environment="Auto", body_proportions="Auto", era="none",
                video_task="Swap", outfit_1="none"):
        def active(values):
            return list(dict.fromkeys(
                str(value).strip() for value in values
                if not _inactive(value) and str(value).strip().lower() != "random"))

        genres = active((genre_1,) if str(genre_1).lower() != "random" else ())
        motions = active((motion_style_1,))
        looks = active((look_1,))
        lines = []
        for label, values in (
            ("Genre", genres), ("Motion style", motions), ("Visual look", looks),
            ("Dialogue", active((dialogue,))), ("Wardrobe", active((wardrobe,))), ("Hair & Makeup", active((hair_makeup,))),
            ("Environment", active((environment,))),
            ("Body proportions", active((body_proportions,))),
        ):
            if values:
                lines.append(label + ": " + " + ".join(values))
        scenario = "" if _inactive(situation) or str(situation).lower() == "random" else str(situation).strip()
        if scenario:
            lines.append("Situation: " + scenario)
        if not _inactive(era) and str(era).strip().lower() != "random":
            lines.append("Era: " + str(era).strip())
        if any(look in SINGLE_CAMERA_LOOKS for look in looks) or any(g in SINGLE_CAMERA_GENRES for g in genres) or any(
                motion in SINGLE_CAMERA_MOTIONS for motion in motions):
            lines.append("Coverage: Single take (the selected look or camera is a single fixed camera)")
        if not _inactive(dialogue_density) and str(dialogue).strip().lower() != "no dialogue":
            lines.append("Dialogue density: " + str(dialogue_density).strip())
        if looks:
            lines.append("Palette: follows the selected Visual look")
        outfits = active((outfit_1,))
        if outfits:
            lines.append("Outfit style: " + " + ".join(outfits))
        if scenario in ADULT_SITUATIONS:
            lines.append("Adults only: every character is a consenting adult (18+)")
        if str(video_task).strip() in VIDEO_TASKS[1:]:
            lines.append("Video task: " + str(video_task).strip())
        drawn = [label for label, value in (("Genre", genre_1), ("Situation", situation),
                                            ("Motion style", motion_style_1), ("Visual look", look_1),
                                            ("Outfit style", outfit_1), ("Era", era))
                 if str(value).strip().lower() == "random"]
        if drawn:  # Random: the director draws one (none: the request alone decides)
            lines.append("Random pick: " + ", ".join(drawn))
        return io.NodeOutput("\n".join(lines))
