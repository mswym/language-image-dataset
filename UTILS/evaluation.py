
from UTILS.utils import *

def cal_evaluation(path, model_name, contexts, labels):
  if not os.path.exists(path+"/DATA/"+model_name+"_benchmark_results.pt"):
    results = {context:{} for context in contexts}
    super_labels = [super_label for super_label in labels]
    basic_labels = sorted(sum([[basic_label for basic_label in labels[super_label]] for super_label in labels],[]))

    print(f"SUPERORDINATE CATEGORIES :\t {super_labels}")
    print(f"BASIC CATEGORIES :\t\t {basic_labels}")

    # -----------------------------------------------------------------------------------------------------------------

    for context in contexts :
      print(f"CONTEXT : {context}[...]")
      text_super = tokenize_fn([context+label for label in super_labels]).to(device)
      text_basic = tokenize_fn([context+label for label in basic_labels]).to(device)

      # ----- COLLECTING THE DATA -------------------------------------------------------------------------------------------------
      original_predictions = compute_original_preds(batch_size=128)
      wordsAdd_predictions = compute_new_preds(batch_size=128)

      # ----- TEST 1 - EFFICIENCY OF WORD-ADDITION (EWA) - 4 NUMBERS --------------------------------------------------------------------------------
      EWA = get_EWA()

      # ----- TEST 2&3 - NEW WORD & ADDED WORD CORRELATION (NAC) - ORIGINAL WORD & ADDED WORD CORRELATION (OAC) - 4 DISTRIBUTIONS --------------------------------------------------------------------------------
      # Ref nonswitch : similarity between original prediction & added-word on nonEFFECTIVE word-added images
      REF_nonswitch_semantic = get_word_correlation_references_nonswitchonly(semantic_similarity_w2v)
      REF_nonswitch_spelling = get_word_correlation_references_nonswitchonly(jellyfish.jaro_winkler_similarity)

      # OAC distributions : similarity between original prediction & added-word on EFFECTIVE word-added images
      OAC_semantic = get_OAC(semantic_similarity_w2v)
      OAC_spelling = get_OAC(jellyfish.jaro_winkler_similarity)

      # NAC distributions : similarity between new prediction & added-word on EFFECTIVE word-added images
      NAC_semantic = get_NAC(semantic_similarity_w2v)
      NAC_spelling = get_NAC(jellyfish.jaro_winkler_similarity)

      # ----- TEST 4 - CONFIDENCE ON MISCLASSIFIED images (COM) - 4 DISTRIBUTIONS ------
      COM_original = get_COM_original()
      COM_new      = get_COM_new()
      REF_nonswitch_proba = get_probabilities_references_nonswitched()
      COM_neworiginal = get_COM_neworiginal()

      results[context] = [EWA, REF_nonswitch_semantic, REF_nonswitch_spelling, NAC_semantic, NAC_spelling, OAC_semantic, OAC_spelling, COM_original, COM_new, REF_nonswitch_proba, COM_neworiginal]

    torch.save(results,path+"/DATA/"+model_name+"_benchmark_results.pt")
  # -----------------------------------------------------------
