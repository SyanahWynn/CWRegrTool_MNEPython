class CWLcorr:
    """ Carbon-wire loop (CWL) based artifact rejection
    This class uses the data from CWLs, 
    which should contain only information about MR-induced ballistocardiogram (BCG) 
    and helium pump artifacts and not brain data, to correct EEG data. 
    It uses a sliding time-window-based linear projection utilizing Hanning tapers. 
    The windows are applied to every channel in both the EEG and the CWL data. 
    To account for small differences in the timing between EEG and CWL channels, 
    the windowed data of the CWLs is delayed embedded. 
    This delay-embeded CWL data is then used as regressors. 
    The channel EEG-data is projected onto this regressor space and coefficients (weights) 
    that best represent the EEG-data as a linear combination of the CWL data are calculated. 
    These weights are then used to create a signal that best represents the EEG data 
    as a linear combination of the CWL data in the least-squares sense; 
    the part of the EEG data that reflects the artifacts. 
    This artifact signal is subsequently subtracted from the EEG data. 
    Leaving the EEG data to represent the variation in the data which is not explained by 
    the CWL data.
    This class is based on: https://github.com/jnvandermeer/CWRegrTool
    """
    def __init__(self, data, cwl_dict):
        """
        Keyword argument:
        cwl_dict            -- A dictionary holding the variables needed for the CWL-based artifact correction
        data                -- MNE data structure
        
        Required keys:
        idx_eeg             -- A list of CWL channel indices; data which needs to be regressed out
        idx_cwl             -- A list of EEG channel indices; data to be corrected

        Optional keys:
        window_duration     -- Window duration in sec. Default = 4
        delay               -- Allowed delay/shift of regressors in sec (recommended to be between 10 and 30 ms). Default = 0.021
        taper_factor        -- Tapering factor. (Tapering factor of 1 means a 50% overlap between consecutive Hann windows); A tapering factor of 2 means 75% overlap between consecutive Hann windows). Default = 1
        taper_function      -- Tapering function (only supporting "hanning" at the moment). Default = "hanning"

        Returns:
        corrected_data      -- EEG data with the CWL-captured artifacts removed
        artifact_data       -- CWL-captured artifacts from the EEG data
        """

        # CHECK IF INPUT DICTIONARY CONTAINS ALL REQUIRED FIELDS
        keys_req = {"idx_eeg", "idx_cwl"} # required keys
        keys_inp = set(cwl_dict.keys()) # keys in the dictionary
        if not keys_req.issubset(keys_inp):
            raise Exception('\nNot all required variables are present. \nPlease check if "idx_eeg" and "idx_cwl" are keys in the struct.')
           
        # USER-DEFINED OR DEFAULT
        self.data       = data
        self.idx_eeg    = cwl_dict["idx_eeg"]
        self.idx_cwl    = cwl_dict["idx_cwl"]
        self.win_dur    = cwl_dict.get("window_duration", 4)
        self.delay      = cwl_dict.get("delay", 0.021)
        self.tap_fac    = cwl_dict.get("taper_factor", 1)
        self.tap_fun    = cwl_dict.get("taper_function", "hanning") 
        
        # CREATED
        # data to apply regression to
        self.x              = data._data[cwl_dict["idx_eeg"],:]
        # data to make the regressors
        self.regs           = data._data[cwl_dict["idx_cwl"],:]
        # number of windows
        self.nwins          = 2 * (2 ** cwl_dict["taper_factor"] - 1) + 1
        # number of samples in a window
        self.win_nsamps     = math.floor(data.info["sfreq"]*cwl_dict["window_duration"] + 1)
        # overlap in consecutive  tapers
        self.tap_ovl        = 1 - (1 / 2 ** cwl_dict["taper_factor"])
        # step size of the tapers in samples
        self.tap_stp_samp   = int(((math.floor(data.info["sfreq"]*cwl_dict["window_duration"] + 1)) - 1) / (2 ** cwl_dict["taper_factor"]))
        # determine taper step duration in samples
        self.tap_dly_samp   = math.floor(data.info["sfreq"]*cwl_dict["delay"])

    def make_window(self):
        """ Create the window """
        if "hann" in self.tap_fun.lower():
            self.window          = np.hanning(self.win_nsamps)
        else:
            raise Exception("Taper function not supported, please specify 'Hanning'")

    @staticmethod
    def delay_embed_single_signal(x, k, step=1, shift=0):
        """ Delays-embed a single signal.

        Parameters:
        x : array-like
            Input signal.
        k : int
            Embedding dimension.
        step : int, optional
            Embedding delay. Default is 1.
        shift : int, optional
            Embedding shift. Default is 0.

        Returns:
        y : np.ndarray
            Delay-embedded reconstructed state space, i.e., a KxT matrix.
        """
        n = x.shape[0]
        embed_dim = k * n
        embed_sample_width = (k - 1) * step + 1
        extra_sample = shift + embed_sample_width - 1
        extra_sample_l = extra_sample // 2
        extra_sample_r = extra_sample - extra_sample_l

        x_padded = np.hstack((np.fliplr(x[:, :extra_sample_l]), x, np.fliplr(x[:, -extra_sample_r:])))
        embed_samples = x_padded.shape[1] - shift - embed_sample_width + 1

        y = np.empty((embed_dim, embed_samples), dtype=x.dtype)

        for j in range(k):
            s = shift + step * j
            y[j * n:(j + 1) * n, :] = x_padded[:, s:s + embed_samples]

        y = np.flipud(y)

        return y

    def delay_embed(self, X, k, step=1, shift=0):
        """ Delays-embed a signal or list of signals.

        Parameters:
        X : array-like or list of array-like
            Input signal(s).
        k : int
            Embedding dimension.
        step : int, optional
            Embedding delay. Default is 1.
        shift : int, optional
            Embedding shift. Default is 0.

        Returns:
        Y : np.ndarray or list of np.ndarray
            Delay-embedded reconstructed state space(s), i.e., a KxT matrix or list of such matrices.
        
        based on: https://github.com/jnvandermeer/CWRegrTool/blob/master/%2Bmisc/delay_embed.m
        """
        if X is None or k is None:
            raise ValueError("Not enough input arguments")

        # Deal with the case of multiple input signals
        if isinstance(X, list):
            return [self.delay_embed_single_signal(x, k, step, shift) for x in X]

        if not isinstance(k, int) or k <= 0:
            raise ValueError('The embedding dimension must be a positive integer')
        if not isinstance(shift, int):
            raise ValueError('The embedding shift must be an integer')
        if not isinstance(step, int) or step <= 0:
            raise ValueError('The embedding delay must be a positive integer')

        return self.delay_embed_single_signal(X, k, step, shift)

    def get_art_signal(self):
        """ Get an artifact signal that best represents the EEG-data as a linear combination of the CWL data and return these"""
        # run the function to create the window
        self.make_window()

        # variables
        win_nsamps = self.win_nsamps
        n_tapers = 2 ** self.tap_fac
        num_eeg_channels = len(self.idx_eeg)
        num_samples = np.shape(self.x)[1]

        # make sure that the boundaries of the tapering windows always falls on a sample
        while ((win_nsamps-1) / n_tapers) % 1 > 0:
            win_nsamps = win_nsamps + 1
            
        # make placeholders
        matrix_stored_fits          = np.zeros((num_eeg_channels, win_nsamps, self.nwins)) # fitted regressors
        matrix_stored_weights       = np.zeros((win_nsamps, self.nwins)) # weights of fitted regressors
        subtracted_signals          = np.zeros((np.shape(self.x))) # substracted data
        subtracted_signals_weights  = np.zeros((num_samples,1)) # weights of the substracted data

        # while loop setup
        current_sample  = 0
        cnt = 0
        t_start = time.time()
        tap_stp_samp = self.tap_stp_samp
        
        while current_sample < num_samples - win_nsamps:
            ## SETUP
            # initiate counter
            cnt         += 1
            cnt_tot     = round(num_samples / tap_stp_samp) - n_tapers
            t_diff      = round((time.time() - t_start) / 60, 2)

            message = f"Window {cnt} out of {cnt_tot}. Elapsed time: {t_diff} min. "
            if cnt == 1:
                message += "Remaining time: Unknown"
            else:
                t_est = round((t_diff / (cnt - 1)) * (cnt_tot - (cnt - 1)), 2)
                message += f"Predicted remaining time: {t_est} min"
            
            # Print the message with carriage return to overwrite previous output
            sys.stdout.write(message)
            sys.stdout.flush()
            
            ## CREATE DATA TO BE SUBTRACTED
            if current_sample >= win_nsamps-1:
                # make placeholders
                summation                   = np.zeros((num_eeg_channels, tap_stp_samp+1)) # summation over windows
                summation_weights           = np.zeros((tap_stp_samp+1,1)) # summation weights over windows
                
                # loop over the overlapping windows, set by the tapering factor
                for i in range(1, n_tapers + 1):
                    # depending on the overlap in windows (determined by the taper factor (2^taper factor))
                    # sum the windowed CWL-regression fitted EEG data
                    # i.e., from the previous (most recent) window, take the first part, from the window before that, take the second part, from the window before that, takke the third part, and so on, etc, etc... until you have summed it all nicely.
                    sumrange = np.arange((i-1) * tap_stp_samp, i * tap_stp_samp+1) + 1
                    summation += matrix_stored_fits[:, sumrange -1, i-1]
                    summation_weights += matrix_stored_weights[sumrange - 1, i-1].reshape(-1, 1)
                
                # substract the summed CWL-regression fitted EEG data from the EEG data
                # so that the signal from the EEG channel wich reflects artifacts captured in the CWL are removed from the EEG data
                subtractrange = np.arange(current_sample - tap_stp_samp + 1, current_sample + 1)
                subtracted_signals[:, subtractrange] = summation[:, 1:]
                subtracted_signals_weights[subtractrange] = summation_weights[1:]

            ## FIT REGRESSION (CWL) MATRIX TO SIGNAL (EEG) MATRIX
            # this is done to 
            # select the range of the window from the data
            xpart           = self.x[:, current_sample:current_sample+win_nsamps]
            regspart        = self.regs[:, current_sample:current_sample+win_nsamps]
            # time-expand the regressors (delayed embedding)
            expregs         = np.flipud(self.delay_embed(regspart,(1+2*self.tap_dly_samp)))
            # taper it with the predefined window
            if len(self.window)!=0:
                expregs = (self.window[:, np.newaxis] * expregs.T).T
            # calculate the inverse of the matrix
            inv_expregs     = scipy.linalg.pinv(expregs)
            
            # make placeholders
            fittedregs      = np.zeros((np.shape(xpart))) # EEG data, fitted with the regressors
            # loop over the EEG channels
            for i_x in range(0, xpart.shape[0]): 
                # taper the data from this channel (or not)
                datavec            = xpart[i_x, :] * self.window if len(self.window) != 0 else xpart[i_x, :]
                # fit the tapered EEG channel data to the model/regressors defined by the tapered and 
                # delay-embedded CWL data and then reconstruct the EEG channel data from this fit.
                # This fit represents the fit between the EEG data and the CWL data and thus 
                # should reflect the artifacts only
                fittedregs[i_x,:]   = datavec.dot(inv_expregs).dot(expregs)

            ## FIFO (FIRST-IN-FIRST-OUT) RULE TO SHIFT WINDOWS BACKWARD
            # Shift the windows backwards using the FIFO rule
            matrix_stored_fits[:, :, 1:] = matrix_stored_fits[:, :, :-1]
            matrix_stored_weights[:, 1:] = matrix_stored_weights[:, :-1]

            # Insert the new data at the first position
            matrix_stored_fits[:, :, 0] = fittedregs
            matrix_stored_weights[:, 0] = self.window

            # COUNT UPDATE
            current_sample += tap_stp_samp

            # Use carriage return (\r) to overwrite the current line in the console
            sys.stdout.write('\r')

        ## DATA TO BE SUBTRACTED ADJUSTMENT
        # not sure why this is here, but it is not doing much..
        inds = subtracted_signals_weights[:,0] > 0
        subtracted_signals[:, inds] = subtracted_signals[:, inds] / np.ones((self.x.shape[0], 1)).dot(subtracted_signals_weights[inds].T)

        # STORE THE RELEVANT VARIABLES
        self.art_sig_w = subtracted_signals_weights
        self.art_sig = subtracted_signals
        self.fits = matrix_stored_fits
        self.weights = matrix_stored_weights

        # return the data
        return self.art_sig
    
    def get_clean_signal(self):
        """ Subtracts the artifact signal created in get_clean_signal() from the EEG data and returns the clean data """
        print("\nGET CWL-CLEANED DATA\n")
        # if needed get the artifact signal
        if not hasattr(self, "art_sig"):
            self.get_art_signal()
        
        ## SUBTRACT THE ARTIFACT SIGNAL FROM THE EEG DATA
        data_clean = self.data.copy()
        data_clean._data[self.idx_eeg,:] -= self.art_sig

        # return the data
        return data_clean
