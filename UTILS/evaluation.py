from UTILS.utils import *
import jellyfish

def cal_evaluation(path, device, model, preprocess, images, model_name, contexts, labels, tokenize_fn):
  if not os.path.exists(path+"DATA/"+model_name+"_benchmark_results.pt"):
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
      original_predictions = compute_original_preds(device, model, preprocess, images, text_basic, text_super, model_name, contexts, context, batch_size=128)
      wordsAdd_predictions = compute_new_preds(device, model, preprocess, images, super_labels, basic_labels, text_basic, text_super, model_name, contexts, context, batch_size=128)

      # ----- TEST 1 - EFFICIENCY OF WORD-ADDITION (EWA) - (task switching in the paper)) --------------------------------------------------------------------------------
      EWA = get_EWA(super_labels, basic_labels, original_predictions, wordsAdd_predictions)

      # Ref nonswitch : semantic/spelling similarity between original prediction & added-word on nonEFFECTIVE word-added images (fig3)
      REF_nonswitch_semantic = get_word_correlation_references_nonswitchonly(semantic_similarity_w2v, basic_labels, super_labels, original_predictions, wordsAdd_predictions)
      REF_nonswitch_spelling = get_word_correlation_references_nonswitchonly(jellyfish.jaro_winkler_similarity, basic_labels, super_labels, original_predictions, wordsAdd_predictions)

      # OAC distributions : semantic/spelling similarity between original prediction & added-word on EFFECTIVE word-added images (fig3)
      OAC_semantic = get_OAC(semantic_similarity_w2v, basic_labels, super_labels, original_predictions, wordsAdd_predictions)
      OAC_spelling = get_OAC(jellyfish.jaro_winkler_similarity, basic_labels, super_labels, original_predictions, wordsAdd_predictions)

      # ----- TEST 4 - CONFIDENCE ON MISCLASSIFIED images (COM) - 4 DISTRIBUTIONS (figure e1)------
      COM_original = get_COM_original(basic_labels, super_labels, original_predictions, wordsAdd_predictions)
      COM_new      = get_COM_new(basic_labels, super_labels, original_predictions, wordsAdd_predictions)
      REF_nonswitch_proba = get_probabilities_references_nonswitched(basic_labels, super_labels, original_predictions, wordsAdd_predictions)
      COM_neworiginal = get_COM_neworiginal(basic_labels, super_labels, original_predictions, wordsAdd_predictions)

      results[context] = [EWA, REF_nonswitch_semantic, REF_nonswitch_spelling, OAC_semantic, OAC_spelling, COM_original, COM_new, REF_nonswitch_proba, COM_neworiginal]

    torch.save(results,path+"DATA/"+model_name+"_benchmark_results.pt")
  # -----------------------------------------------------------
