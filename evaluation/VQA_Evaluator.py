import sys
sys.path.insert(0, '/auto/homes/bat34/VQA/PythonHelperTools/vqaTools')
sys.path.insert(0, '/auto/homes/bat34/VQA/PythonEvaluationTools')
from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval

class VQA_Evaluator:
    def __init__(self,
                 summary_writer = None,
                 dataDir='/auto/homes/bat34/VQA',
                 versionType='v2_',
                 taskType='OpenEnded',
                 dataType='mscoco', 
                 dataSubType ='val2014'):
        self.writer = summary_writer
        self.versionType = versionType
        self.taskType = taskType
        self.dataType = dataType
        self.dataSubType = dataSubType
        self.annFile = '%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
        self.quesFile='%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)
        self.vqa = VQA(self.annFile, self.quesFile)
        
    def evaluate(self, resFile, epoch):
        vqaRes = self.vqa.loadRes(resFile, self.quesFile)
        
        # create vqaEval object by taking vqa and vqaRes
        vqaEval = VQAEval(self.vqa, vqaRes, n=2)
        vqaEval.evaluate()
        print("\n")
        print("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))
        if self.writer:
            self.writer.add_scalar('validation/overall_accuracy', vqaEval.accuracy['overall'], epoch)
            
        print("Per Question Type Accuracy is the following:")
        for quesType in vqaEval.accuracy['perQuestionType']:
            print("%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType]))
            if self.writer:
                self.writer.add_scalar('validation/%s' % quesType, vqaEval.accuracy['perQuestionType'][quesType], epoch)
        print("\n")
        print("Per Answer Type Accuracy is the following:")
        for ansType in vqaEval.accuracy['perAnswerType']:
            print("%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            if self.writer:
                self.writer.add_scalar('validation/%s' % ansType, vqaEval.accuracy['perAnswerType'][ansType], epoch)
        print("\n")
        return float(vqaEval.accuracy['overall'])
        
