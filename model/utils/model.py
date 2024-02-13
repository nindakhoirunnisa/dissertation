import torch
import warnings
warnings.filterwarnings("ignore")
import functools
# import tensorflow as tf

class Model(object):
  def __init__(self, config, network):
    self.config = config
    self.network = network
  
  def forward_word_classification(model, batch_data, i2w, is_test=False, device='cpu', **kwargs):
    #unpack batch data
    if len(batch_data) == 4:
      (subword_batch, mask_batch, subword_to_word_indices_batch, label_batch) = batch_data
      token_type_batch = None
    elif len(batch_data) == 5:
      (subword_batch, mask_batch, token_type_batch, subword_to_word_indices_batch, label_batch) = batch_data
    
    #prepare input and model
    subword_batch = torch.LongTensor(subword_batch)
    mask_batch = torch.FloatTensor(mask_batch)
    token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
    subword_to_word_indices_batch = torch.LongTensor(subword_to_word_indices_batch)
    label_batch = torch.LongTensor(label_batch)

    if device == 'cuda':
      subword_batch = subword_batch.cuda()
      mask_batch = mask_batch.cuda()
      token_type_batch = token_type_batch.cuda() if token_type_batch is not None else None
      subword_to_word_indices_batch = subword_to_word_indices_batch.cuda()
      label_batch = label_batch.cuda()
    
    #forward model
    outputs = model(subword_batch, subword_to_word_indices_batch, attention_mask=mask_batch, token_type_ids=token_type_batch, labels=label_batch)
    # loss, logits = outputs[:2] #bilstm
    loss, crf = outputs[:2] #crf

    #generate prediction & label list
    list_hyps = []
    list_labels = []
    # hyps_list = torch.topk(logits, k=1, dim=-1)[1].squeeze(dim=-1)
    # for i in range(len(hyps_list)):
    for i in range(len(crf)):
      hyps, labels = crf[i], label_batch[i].tolist()      
      # hyps, labels = hyps_list[i].tolist(), label_batch[i].tolist()        
      list_hyp, list_label = [], []
      for j in range(len(hyps)):
        if labels[j] == -100:
          break
        else:
          list_hyp.append(i2w[hyps[j]])
          list_label.append(i2w[labels[j]])
      list_hyps.append(list_hyp)
      list_labels.append(list_label)
          
    return loss, list_hyps, list_labels

  def evaluate(model, data_loader, forward_fn, metrics_fn, i2w, is_test=False):
      model.eval()
      total_loss, total_correct, total_labels = 0, 0, 0

      list_hyp, list_label, list_seq = [], [], []

      pbar = tqdm(iter(data_loader), leave=True, total=len(data_loader))
      for i, batch_data in enumerate(pbar):
          batch_seq = batch_data[-1]        
          loss, batch_hyp, batch_label = forward_fn(model, batch_data[:-1], i2w=i2w, device=args['device'])

          
          # Calculate total loss
          test_loss = loss.item()
          total_loss = total_loss + test_loss

          # Calculate evaluation metrics
          list_hyp += batch_hyp
          list_label += batch_label
          list_seq += batch_seq
          metrics = metrics_fn(list_hyp, list_label)

          if not is_test:
              pbar.set_description("VALID LOSS:{:.4f} {}".format(total_loss/(i+1), metrics_to_string(metrics)))
          else:
              pbar.set_description("TEST LOSS:{:.4f} {}".format(total_loss/(i+1), metrics_to_string(metrics)))
      
      if is_test:
          return total_loss, metrics, list_hyp, list_label, list_seq
      else:
          return total_loss, metrics

  # Training function and trainer
  def train(model, train_loader, valid_loader, optimizer, forward_fn, metrics_fn, valid_criterion, i2w, n_epochs, evaluate_every=1, early_stop=3, step_size=1, gamma=0.5, model_dir="", exp_id=None):
      scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

      best_val_metric = -100
      count_stop = 0

      for epoch in range(n_epochs):
          model.train()
          total_train_loss = 0
          list_hyp, list_label = [], []
          
          train_pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
          for i, batch_data in enumerate(train_pbar):
              # forward_fn is used to get loss, hyp, and labels
              loss, batch_hyp, batch_label = forward_fn(model, batch_data[:-1], i2w=i2w, device=args['device'])
              # if i == 1:
              #   print("hyp", batch_hyp)
              #   print("label", batch_label)

              optimizer.zero_grad()
              if args['fp16']:
                  with amp.scale_loss(loss, optimizer) as scaled_loss:
                      scaled_loss.backward()
                  torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_norm'])
              else:
                  loss.backward()
                  torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_norm'])
              optimizer.step()

              tr_loss = loss.item()
              total_train_loss = total_train_loss + tr_loss

              # Calculate metrics
              list_hyp += batch_hyp
              list_label += batch_label
              
              train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
                  total_train_loss/(i+1), get_lr(args, optimizer)))
                          
          metrics = metrics_fn(list_hyp, list_label)
          print("(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format((epoch+1),
              total_train_loss/(i+1), metrics_to_string(metrics), get_lr(args, optimizer)))
          
          # Decay Learning Rate
          scheduler.step()

          # evaluate
          if ((epoch+1) % evaluate_every) == 0:
              val_loss, val_metrics = evaluate(model, valid_loader, forward_fn, metrics_fn, i2w, is_test=False)

              # Early stopping
              val_metric = val_metrics[valid_criterion]
              if best_val_metric < val_metric:
                  best_val_metric = val_metric
                  # save model
                  if exp_id is not None:
                      torch.save(model.state_dict(), model_dir + "/best_model_" + str(exp_id) + ".th")
                  else:
                      torch.save(model.state_dict(), model_dir + "/best_model.th")
                  count_stop = 0
              else:
                  count_stop += 1
                  print("count stop:", count_stop)
                  if count_stop == early_stop:
                      break

#Need to make function: predict, predict_tags, predict_viterbi_score, predict_probs, predict_logits, predict_mnlp_score
