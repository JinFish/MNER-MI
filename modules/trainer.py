import torch
from torch import optim
from tqdm import tqdm
from seqeval.metrics import classification_report 
from transformers.optimization import get_linear_schedule_with_warmup
from ner_evaluate import evaluate
from torch.utils import data

class NERTrainer:
    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None, label_map=None,
                 args=None, logger=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model 
        self.logger = logger
        self.label_map = label_map
        self.refresh_step = 2
        self.best_train_metric = 0
        self.best_train_epoch = 0
        self.best_dev_metric = 0
        self.best_dev_epoch = 0
        self.optimizer = None

        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
        self.step = 0
        self.args = args

    def train(self):
        self.bert_before_train()
        
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        
        pbar = tqdm(total=self.train_num_steps)
        avg_loss = 0  
        
        for epoch in range(1, self.args.num_epochs+1):
            y_true, y_pred = [], []
            y_true_idx, y_pred_idx = [], []
            
            pbar.set_description(f"Epoch {epoch}/{self.args.num_epochs+1}")
            self.model.train()
            
            for batch in self.train_data:
                self.step += 1
                batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                attention_mask, labels, logits, loss = self._step(batch)
                avg_loss += loss.detach().cpu().item()

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if isinstance(logits, torch.Tensor):  # CRF return lists
                    logits = logits.argmax(-1).detach().cpu().numpy()  # batch, seq, 1
                label_ids = labels.to('cpu').numpy()
                input_mask = attention_mask.to('cpu').numpy()
                label_map = {idx: label for label, idx in self.label_map.items()}
                reverse_label_map = {label: idx for label, idx in self.label_map.items()}

                for row, mask_line in enumerate(input_mask):
                    true_label = []
                    true_label_idx = []
                    true_predict = []
                    true_predict_idx = []
                    for column, mask in enumerate(mask_line):
                        if column == 0:
                            continue
                        if mask:
                            if label_map[label_ids[row][column]] != "X":
                                true_label.append(label_map[label_ids[row][column]])
                                true_label_idx.append(label_ids[row][column])
                                true_predict.append(label_map[logits[row][column]])
                                true_predict_idx.append(logits[row][column])
                        else:
                            break
                    y_true.append(true_label)
                    y_true_idx.append(true_label_idx)
                    y_pred.append(true_predict)
                    y_pred_idx.append(true_predict_idx)

                if self.step % self.refresh_step == 0:
                    avg_loss = float(avg_loss) / self.refresh_step
                    pbar.set_postfix(loss=avg_loss)
                    avg_loss = 0
                    pbar.update(self.refresh_step)
            
            results = classification_report(y_true, y_pred, digits=4)
            acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, y_pred, reverse_label_map)

            self.logger.info("***** Train Eval results *****")
            self.logger.info("\n%s", results)
            f1_score = float(results.split('\n')[-4].split('      ')[0].split('    ')[3])

            if f1_score > self.best_train_metric:
                self.best_train_metric = f1_score
                self.best_train_epoch = epoch

            self.logger.info(f"Epoch {epoch}/{self.args.num_epochs}, best train f1: {self.best_train_metric},\
                            best train epoch: {self.best_train_epoch}, current train f1 score: {f1_score}")

            self.evaluate(epoch)  # generator to dev.

        pbar.close()
        self.pbar = None
        self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch,
                                                                                                self.best_dev_metric))
        self.logger.info("The best max_f1 = %s", str(self.best_dev_metric))

        self.test()


    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        y_true, y_pred = [], []
        y_true_idx, y_pred_idx = [], []
        step = 0

        with torch.no_grad():
            self.model.eval()
            total_loss = 0
            for batch in self.dev_data:
                step += 1
                batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in
                            batch)  # to cpu/cuda device
                attention_mask, labels, logits, loss = self._step(batch)  # logits: batch, seq, num_labels
                total_loss += loss.detach().cpu().item()

                if isinstance(logits, torch.Tensor):
                    logits = logits.argmax(-1).detach().cpu().numpy()  # batch, seq, 1
                label_ids = labels.detach().cpu().numpy()
                input_mask = attention_mask.detach().cpu().numpy()
                label_map = {idx: label for label, idx in self.label_map.items()}
                reverse_label_map = {label: idx for label, idx in self.label_map.items()}
                for row, mask_line in enumerate(input_mask):
                    true_label = []
                    true_label_idx = []
                    true_predict = []
                    true_predict_idx = []
                    for column, mask in enumerate(mask_line):
                        if column == 0:
                            continue
                        if mask:
                            if label_map[label_ids[row][column]] != "X":
                                true_label.append(label_map[label_ids[row][column]])
                                true_label_idx.append(label_ids[row][column])
                                true_predict.append(label_map[logits[row][column]])
                                true_predict_idx.append(logits[row][column])
                        else:
                            break
                    y_true.append(true_label)
                    y_true_idx.append(true_label_idx)
                    y_pred.append(true_predict)
                    y_pred_idx.append(true_predict_idx)

            results = classification_report(y_true, y_pred, digits=4)
            acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, y_pred, reverse_label_map)
            self.logger.info("***** Dev Eval results *****")
            self.logger.info("\n%s", results)
            f1_score = float(results.split('\n')[-4].split('      ')[-2].split('    ')[-1])

            if f1_score >= self.best_dev_metric:  # this epoch get best performance
                self.logger.info("Get better performance at epoch {}".format(epoch))
                self.best_dev_epoch = epoch
                self.best_dev_metric = f1_score  # update best metric(f1 score)
                torch.save(self.model.state_dict(), "./model.pt")

            self.logger.info(f"Epoch {epoch}/{self.args.num_epochs}, best dev f1: {self.best_dev_metric},\
                            best dev epoch: {self.best_dev_epoch}, current dev f1 score: {f1_score}")

            
        self.model.train()
        
    def test(self):
        self.model.load_state_dict(torch.load("./model.pt"))
        self.model.to(self.args.device)
        self.model.eval()
        self.logger.info("***** Running test *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        y_true, y_pred = [], []
        y_true_idx, y_pred_idx = [], []
        step = 0

        with torch.no_grad():
            self.model.eval()
            total_loss = 0
            for batch in self.test_data:
                step += 1
                batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in
                            batch)  # to cpu/cuda device
                attention_mask, labels, logits, loss = self._step(batch)  # logits: batch, seq, num_labels
                total_loss += loss.detach().cpu().item()

                if isinstance(logits, torch.Tensor):
                    logits = logits.argmax(-1).detach().cpu().numpy()  # batch, seq, 1
                label_ids = labels.detach().cpu().numpy()
                input_mask = attention_mask.detach().cpu().numpy()
                label_map = {idx: label for label, idx in self.label_map.items()}
                reverse_label_map = {label: idx for label, idx in self.label_map.items()}
                for row, mask_line in enumerate(input_mask):
                    true_label = []
                    true_label_idx = []
                    true_predict = []
                    true_predict_idx = []
                    for column, mask in enumerate(mask_line):
                        if column == 0:
                            continue
                        if mask:
                            if label_map[label_ids[row][column]] != "X":
                                true_label.append(label_map[label_ids[row][column]])
                                true_label_idx.append(label_ids[row][column])
                                true_predict.append(label_map[logits[row][column]])
                                true_predict_idx.append(logits[row][column])
                        else:
                            break
                    y_true.append(true_label)
                    y_true_idx.append(true_label_idx)
                    y_pred.append(true_predict)
                    y_pred_idx.append(true_predict_idx)

            results = classification_report(y_true, y_pred, digits=4)
            acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, y_pred, reverse_label_map)
            self.logger.info("***** Test results *****")
            f1_score = float(results.split('\n')[-4].split('      ')[-2].split('    ')[-1])


            self.logger.info(f"--- Test f1 score is {f1_score} ---")



    def _step(self, batch):
        input_ids, token_type_ids, attention_mask, image_features, labels = batch

        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                            img_features=image_features, labels=labels)
        logits, loss = output
        return attention_mask, labels, logits, loss
        

    def bert_before_train(self):
        parameters = []
        parameters = []
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            params['params'].append(param)
        parameters.append(params)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)

        self.model.to(self.args.device)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)
