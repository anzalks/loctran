import logging
logging.basicConfig(level=logging.DEBUG)

from loctran.translate import translate_segments

segments = [
    {"id": 0, "text": "Loctran - Traducteur de PDF Local et Prive"},
    {"id": 1, "text": "Loctran est un logiciel libre de traduction de documents PDF qui fonctionne\nentierement en local, sans connexion internet ni cle API. Il exploite Ollama\npour acceder a des modeles de langage open source, garantissant ainsi une\nconfidentialite totale de vos documents sensibles."},
    {"id": 2, "text": "Fonctionnalites principales :"},
    {"id": 3, "text": "1. Confidentialite totale\n   Vos fichiers ne quittent jamais votre machine. Loctran communique\n   uniquement avec Ollama en local sur localhost:11434. Aucune telemetrie,\n   aucune analyse, aucun nuage."},
    {"id": 4, "text": "2. OCR avance double passe\n   Tesseract effectue une double passe OCR (image normale + image inversee)\n   pour detecter le texte clair sur fond sombre ou a faible contraste.\n   Les modeles de vision comme glm-ocr offrent une precision superieure."},
    {"id": 5, "text": "3. Interface web intuitive\n   Un tableau de bord web avec suivi en temps reel permet de televerser un\n   PDF, de choisir la langue cible et le modele, puis de suivre chaque etape\n   du traitement grace a une barre de progression interactive."},
    {"id": 6, "text": "4. Sortie HTML superposee\n   Le resultat est un fichier HTML ou les traductions sont positionnees\n   exactement a l'emplacement du texte original dans le document, preservant\n   ainsi la mise en page d'origine."},
    {"id": 7, "text": "5. Compression PDF integree\n   Loctran inclut egalement un outil de compression PDF permettant de\n   reduire la taille des fichiers sans dependances proprietaires."},
    {"id": 8, "text": "Installation rapide :\n    pip install loctran\n    ollama pull glm-ocr\n    ollama pull translategemma:4b\n    loctran"},
    {"id": 9, "text": "Ce document illustre les capacites de Loctran en matiere de traduction.\nIl sera traduit vers l'anglais, demontrant le pipeline complet du logiciel."},
]

results = translate_segments(segments, model="translategemma:4b", target_lang="en")

for i, s in enumerate(segments):
    print(f"\n--- Segment {i} ---")
    print("ORIGINAL:", s["text"])
    print("TRANSLATION:", results.get(i))
