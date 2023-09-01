#' logistic_regr
#'  Run logistic regression on a dataframe.
#'
#' @param df - a dataframe with numeric columns. See requirements below.
#' @param test_pct - the proportion of df used for testing.
#' @param opt_cutoff - the cutoff value for the repsonse to be 1.
#'                     if NULL, uses InformationValue::optimalCutoff
#'
#' @return
#'  a list
#'    model - the logistic regression model.
#'    prediction - predictions of response the test data on the model.
#'    y_eval - predictions on 0/1 basis. Values >= optimal_cutoff = 1
#'    formula - the formula used for the glm model.
#'    optimal_cutoff - probability cutoff score, based on minimizing 
#'                     test case misclassification.
#'    misclass_error - the proportion of the test cases misclassified.
#'    train - the training data from df.
#'    test - the test data from df.
#'    
#' @requires
#'  All dataframe columns must be numeric.
#'  Target variable must be 0/1 and in last column of df.
#'  InformationValue library from https://github.com/selva86/InformationValue.
#'  If you don't want to use this library, set opt_cutoff to some value,
#'  0 < opt_cutoff < 1.
#'
logistic_regr <- function(df, 
                          test_pct = 0.33,
                          opt_cutoff = NULL) {
  if(is.null(opt_cutoff)) {
    require(InformationValue)
  }
  
  samples <- sample(c(TRUE, FALSE), nrow(df), 
                    replace=TRUE, 
                    prob = c(1-test_pct, test_pct))
 
  train <- df[samples, ]
  test <- df[!samples, ]
  
  # create a formula from the df columns
  formula <- paste(names(train)[ncol(train)], '~', names(train)[1])
  for(col in 2:(ncol(train)-1)) {
    formula <- paste(formula, '+', names(train)[col])
  }
  
  model <- glm(formula, 
               data = train, 
               family = binomial,
               maxit = 100)
  
  prediction <- predict(model, test, type = 'response')
  
  if(is.null(opt_cutoff)) {
    opt_cutoff <- optimalCutoff(test[ncol(test)], prediction)
  }
  
  y_eval <- ifelse(prediction >= opt_cutoff, 1, 0)
  
  # do it this way because InformationValue::misClassError sometimes returns
  # count instead of proportion
  err <- sum(test[, ncol(test)] != y_eval) / nrow(test)
  
  return(list(model = model, 
              prediction = prediction,
              y_eval = y_eval,
              formula = formula,
              optimal_cutoff = opt_cutoff,
              misclass_error = err,
              train = train,
              test = test))
}